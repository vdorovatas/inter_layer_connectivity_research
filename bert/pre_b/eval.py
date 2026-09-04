import json
from pathlib import Path
from typing import Iterator
import time
from trainer import CustomTrainer
import datasets
import matplotlib.pyplot as plt
import pandas as pd
import pynvml
import torch
from magic_timer import MagicTimer
from tokenizers import BertWordPieceTokenizer, Regex, normalizers
from tqdm import tqdm
from model import LCN_BertForMaskedLM
from accelerate import Accelerator
from safetensors.torch import load_model
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import argparse
from metrics import mlm_accuracy, nsp_accuracy, my_metrics #, perplexity

def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT Masked Language Model")
    parser.add_argument("--lcn", type=bool, default=False, help="LCN or Res")
    return parser.parse_args()

def main():

    args = parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    accelerator = Accelerator()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
    print(f"GPU: {gpu_name}, {gpu_mem} MiB")
    print(f"{torch.cuda.is_available() = }")

    ###################################################################

    LIMIT_DATASET = None #2016 * 4  # keep small for development, set to None for full dataset

    RANDOM_SEED = 42
    NUM_TOKENIZER_TRAINING_ITEMS = 1_000_000  # I made this up, but it seems reasonable
    VOCAB_SIZE = 32_768  # from Cramming
    DEVICE_BATCH_SIZE = 32  # adjust to get near 100% gpu memory use
    MODEL_MAX_SEQ_LEN = 128#128  # from Cramming

    gradient_accumulation_steps = 2048 // DEVICE_BATCH_SIZE  # roughly based on Cramming
    batch_size = DEVICE_BATCH_SIZE * gradient_accumulation_steps
    print(f"{DEVICE_BATCH_SIZE = }")
    print(f"{gradient_accumulation_steps = }")
    print(f"{batch_size = }")

    ###################################################################

    MODEL_DIR = 'checkpoints/checkpoint-6959/model.safetensors' 
    TOKENIZER_PATH = 'checkpoints/tokenizer.json'


    ###################################################################
    if LIMIT_DATASET:
        print('\nDEVSET\n')
        with MagicTimer() as timer:
            dataset = datasets.load_dataset(
                "sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen",
                split=f"train[:{LIMIT_DATASET}]",
                revision="0e6fada2dd43136e4a3f637da41de2e596aee674",
            )
        print(f"Loaded dataset in {timer}")

        # Now split the dataset: 80% for training and 20% for evaluation
        with MagicTimer() as timer:
            train_size = 0.9
            eval_size = 0.1
            # Create the training and evaluation splits
            train_dataset = dataset.select(range(int(len(dataset) * train_size))) 
            val_dataset = dataset.select(range(int(len(dataset) * train_size), len(dataset)))  
        print(f"Split dataset into training and evaluation in {timer}")
    else:
        print('\nFULL DATASET\n')
        with MagicTimer() as timer:
            dataset = datasets.load_dataset(
                "sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen",
                split="train",  # Load the entire training portion
                revision="0e6fada2dd43136e4a3f637da41de2e596aee674",
            )
        print(f"Loaded dataset in {timer}")
    
        with MagicTimer() as timer:
            portion_size = 0.7  # 70% of the dataset
            dataset = dataset.shuffle(seed=42)  # Shuffle the dataset
            dataset = dataset.select(range(int(len(dataset) * portion_size)))  # Select 70% of the dataset
        print(f"Selected 70% of the dataset in {timer}")

        # Now split the dataset: 80% for training and 20% for evaluation
        with MagicTimer() as timer:
            train_size = 0.85
            eval_size = 0.15
            # Create the training and evaluation splits
            train_dataset = dataset.select(range(int(len(dataset) * train_size)))  # First 80% for training
            val_dataset = dataset.select(range(int(len(dataset) * train_size), len(dataset)))  # Last 20% for evaluation
            val_dataset = val_dataset.select(range(int(len(val_dataset)*0.01))) 
        print(f"Split dataset into training and evaluation in {timer}")

    ###################################################################

    tokenizer = BertWordPieceTokenizer()
    tokenizer._tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex("(``|'')"), '"'),
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""),
        ]
    )

    ###################################################################
    ###################################################################

    if args.lcn:
        MODEL_DIR = 'checkpoints/checkpoint-6800/model.safetensors' 
        print('\nUsing LCN...\n')
        from transformers import AutoConfig
        config_path = "permanent_checkpoints/checkpoint-3600/config.json"
        config = AutoConfig.from_pretrained(config_path)
        model = LCN_BertForMaskedLM(config)
        tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
        from safetensors import safe_open

        tensors = {}
        with safe_open(MODEL_DIR, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k) # loads the full tensor given a key

        model_state_dict = model.state_dict()
        for key, tensor in tensors.items():
            if key.startswith('module.'): key = key[7:] 
            if key in model_state_dict:
                if model_state_dict[key].shape == tensor.shape:
                    model_state_dict[key] = tensor
                else:
                    print(f"Shape mismatch for {key}: model shape {model_state_dict[key].shape}, tensor shape {tensor.shape}")
            else:
                print(f"Warning: Key {key} not found in model state_dict")

        model.load_state_dict(model_state_dict)
    else:
        from transformers import BertConfig, BertModel
        from safetensors.torch import load_file
        TOKENIZER_PATH = 'checkpoints_vanilla/tokenizer.json'
        MODEL_DIR = 'checkpoints_vanilla/checkpoint-4200/model.safetensors' 
        config_file_path = 'checkpoints_vanilla/checkpoint-4200/config.json'
        config = BertConfig.from_pretrained(config_file_path)
        weights = load_file(MODEL_DIR)
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        model = BertForMaskedLM(config)
        model.load_state_dict(weights, strict=False)
        tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to('cuda') 

    ###################################################################

    class TokenizedDataset(torch.utils.data.Dataset):
        "This wraps the dataset and tokenizes it, ready for the model"

        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, i):
            return self.tokenizer.encode(
                self.dataset[i]["text"],
                return_tensors="pt",
                truncation=True,
                max_length=MODEL_MAX_SEQ_LEN - 2,
                padding="max_length",
                return_special_tokens_mask=True,
            )[0, ...]


    train_tokenized_dataset = TokenizedDataset(train_dataset, tokenizer)
    val_tokenized_dataset = TokenizedDataset(val_dataset, tokenizer)
    val_tokenized_dataset = TokenizedDataset(val_dataset.select(range(300)) , tokenizer)

    steps_per_epoch = len(train_tokenized_dataset) // batch_size
    print('STEPS PER EPOCH: ', steps_per_epoch)

    ###################################################################

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        return_tensors="pt",
    )

    ###################################################################

    training_args = TrainingArguments(
    learning_rate=1e-3,
    warmup_ratio=0.5,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-9,
    weight_decay=0.01,
    max_grad_norm=0.5,
    num_train_epochs=1,
    eval_strategy='epoch',
    per_device_train_batch_size=DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=gradient_accumulation_steps,
    dataloader_num_workers=4,
    save_steps=0,  # Disable saving model checkpoints
    save_total_limit=0,  # Remove any limit on saved checkpoints
    logging_steps=1,
    output_dir='eval_dir',  # Disable output directory for model saves
    optim="adamw_torch",
    eval_accumulation_steps=16,
    )

    metric_functions = my_metrics 

    CustomTrainer._get_train_sampler = lambda _: None  # prevent shuffling the dataset again
    print('Started...')
    trainer = CustomTrainer(
        model=model,
        lcn=args.lcn,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        compute_metrics=metric_functions
    )

    ###################################################################

    #%env TOKENIZERS_PARALLELISM=false
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with MagicTimer() as timer:
        print('Started validation...')
        validation_results = trainer.evaluate(eval_dataset=val_tokenized_dataset)
    print(f"Evaluated model in {timer}.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()


