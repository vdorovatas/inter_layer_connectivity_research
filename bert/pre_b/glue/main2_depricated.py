from models2 import LCNBertForSequenceClassification
from data2 import prepare
import torch
import numpy as np
import evaluate
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["cola", "mnli", "mnli-mm", "mrpc",
                                                                    "qnli", "qqp", "rte", "sst2", "stsb", "wnli"])

    args = parser.parse_args()
    return args

def run():

    args = parse_args()
    task = args.task
    #GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    actual_task = "mnli" if task == "mnli-mm" else task
    metric = evaluate.load("glue", actual_task)
    batch_size = 16

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    MODEL_DIR = '../checkpoints/checkpoint-6800/model.safetensors' 
    from transformers import AutoConfig
    config_path = "../checkpoints/checkpoint-6800/config.json"
    config = AutoConfig.from_pretrained(config_path)
    model = LCNBertForSequenceClassification(config, num_labels)

    from tokenizers import BertWordPieceTokenizer, Regex, normalizers
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

    TOKENIZER_PATH = '../checkpoints/tokenizer.json'
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


    encoded_dataset = prepare(actual_task, tokenizer)
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to('cuda') 

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = "model"

    optimizer = optim.AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
    epochs = 12
    total_steps = len(encoded_dataset["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    args_ = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy = "epoch",
        save_strategy = "no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        metric_for_best_model=metric_name
    )

    def compute_metrics(eval_pred):
        intermid = []
        preds, labels = eval_pred
        logits, labels, intermid = preds
        print('LEN INTERMID: ', len(intermid))
        if task != "stsb":
            predictions = np.argmax(logits, axis=1)
            intermid_preds = [np.argmax(p, axis=1) for p in intermid]
        else:
            predictions = logits[:, 0]
            intermid_preds = [p[:, 0] for p in intermid]

        m1 = metric.compute(predictions=predictions, references=labels)
        m2 = [metric.compute(predictions=p, references=labels) for p in intermid_preds]
        print("")
        print("Intermid Preds: ", m2)
        print("")
        return m1

    validation_key = "validation" #"validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    trainer = Trainer(
        model,
        args_,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

if __name__ == "__main__":
    run()