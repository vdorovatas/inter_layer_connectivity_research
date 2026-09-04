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
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="directory where checkpoints are stored")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to continue from")
    return parser.parse_args()

def main():

    args = parse_args()

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
    DEVICE_BATCH_SIZE = 64 # adjust to get near 100% gpu memory use
    MODEL_MAX_SEQ_LEN = 128#128  # from Cramming

    gradient_accumulation_steps = 2048 // DEVICE_BATCH_SIZE  # roughly based on Cramming
    batch_size = DEVICE_BATCH_SIZE * gradient_accumulation_steps
    print(f"{DEVICE_BATCH_SIZE = }")
    print(f"{gradient_accumulation_steps = }")
    print(f"{batch_size = }")

    ###################################################################
    if args.checkpoints_dir:
        print('\nFROM CHECKPOINT...\n')
        RUN_DIR = Path(args.checkpoints_dir)
        CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
        print(CHECKPOINT_DIR)
        MODEL_DIR = RUN_DIR / "model"
        TOKENIZER_PATH = RUN_DIR /  "tokenizer.json"
        print(TOKENIZER_PATH)
        TRAINER_HISTORY_PATH = RUN_DIR / "trainer_history.json"
    else:
        print('\nFROM SCRATCH...\n')
        RUN_DIR = Path("runs_final_vanilla") / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
        MODEL_DIR = RUN_DIR / "model"
        TOKENIZER_PATH = RUN_DIR / "tokenizer.json"
        TRAINER_HISTORY_PATH = RUN_DIR / "trainer_history.json"

        RUN_DIR.mkdir(exist_ok=True, parents=True)

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
            portion_size = 1.0  
            dataset = dataset.shuffle(seed=42)  # Shuffle the dataset
            dataset = dataset.select(range(int(len(dataset) * portion_size)))  # Select 70% of the dataset

        # Now split the dataset: 80% for training and 20% for evaluation
        with MagicTimer() as timer:
            train_size = 0.85
            eval_size = 0.15
            # Create the training and evaluation splits
            train_dataset = dataset.select(range(int(len(dataset) * train_size)))  # First 80% for training
            val_dataset = dataset.select(range(int(len(dataset) * train_size), len(dataset)))  # Last 20% for evaluation
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

    def tokenizer_training_data() -> Iterator[str]:
        for i in tqdm(
            range(min(NUM_TOKENIZER_TRAINING_ITEMS, len(train_dataset))),
            desc="Feeding samples to tokenizer",
        ):
            yield train_dataset[i]["text"]

    if args.checkpoints_dir is None:
        
        with MagicTimer() as timer:
            tokenizer.train_from_iterator(
                tokenizer_training_data(),
                vocab_size=VOCAB_SIZE,
                min_frequency=2,
            )
        print(f"Tokenizer trained in {timer}.")
        tokenizer.save(str(TOKENIZER_PATH))

    ###################################################################

    model_config = BertConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MODEL_MAX_SEQ_LEN,
        attention_probs_dropout_prob=0,  # cramming says no dropout
        hidden_dropout_prob=0,  # cramming says no dropout
        output_hidden_states=True,
    )

    if args.lcn:
        print('\nUsing LCN...\n')
        model = LCN_BertForMaskedLM(model_config)
        tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    else:
        model = BertForMaskedLM(model_config)
        tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        print('aaaa: ', model.module._keys_to_ignore_on_save)
        model._keys_to_ignore_on_save = None

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
        # Optimizer values are from Cramming
        learning_rate=1e-3,
        warmup_ratio=0.5,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-9,
        weight_decay=0.01,
        max_grad_norm=0.5,
        num_train_epochs=1,
        #eval_strategy='epoch',
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=4,
        save_steps=200,
        save_total_limit=5,
        logging_steps=1,
        output_dir=CHECKPOINT_DIR,
        optim="adamw_torch",
        #eval_accumulation_steps=16,
    )

    metric_functions = my_metrics #[mlm_accuracy, nsp_accuracy] #, perplexity]

    CustomTrainer._get_train_sampler = lambda _: None  # prevent shuffling the dataset again
    print('Started...')
    trainer = CustomTrainer(
        model=model,
        lcn=args.lcn,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized_dataset,
        #eval_dataset=val_tokenized_dataset,
        #compute_metrics=metric_functions
    )

    ###################################################################

    #%env TOKENIZERS_PARALLELISM=false
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with MagicTimer() as timer:
        if args.checkpoints_dir is None:
            trainer.train()
        elif args.checkpoints_dir and args.checkpoint:
            trainer.train(resume_from_checkpoint=args.checkpoint)
        else:
            print("ERROR")
    print(f"Trained model in {timer}.")
    trainer.save_model(str(MODEL_DIR))
    TRAINER_HISTORY_PATH.write_text(json.dumps(trainer.state.log_history))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

