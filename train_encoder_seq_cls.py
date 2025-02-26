from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
import evaluate
import wandb
import argparse
import json
from utils import compute_average_metrics, get_dataset_length_stats



parser = argparse.ArgumentParser(prog="Sequence Classification Training Script")
parser.add_argument("--model_name", type=str, default="nickprock/sentence-bert-base-italian-xxl-uncased")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()
print(args)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(
    project="PORRO",
    entity="michelej-m",
    name=args.model_name.split("/")[1],
)
wandb.log({"num_runs": args.runs})

results = []

for _ in range(args.runs):
    # ---- Model / Tokenizer loading
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_name.resize_token_embeddings(len(tokenizer))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map=DEVICE,
        num_labels=2,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        #target_modules=["query_proj", "value_proj"],
    )
    model = get_peft_model(model, lora_config)

    # ---- Dataset loading + Processing
    max_len = 512

    dataset = load_dataset(
        "csv", data_files="/home/michele.maggini/PORRO_2/PORRO69_to_release.csv"
    )
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    print(dataset)

    def format_func(element):
        element["labels"] = [0,1].index(element["labels"])
        return element

    def tokenization_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)

    dataset = dataset.map(format_func)
    tokenized_dataset = dataset.map(tokenization_func)

    length_stats = get_dataset_length_stats(tokenizer, dataset)
    print(json.dumps(length_stats, indent=4))

    # ------Training prep
    # -- Hyperparameters
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs

    def compute_metrics(eval_pred):
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
            "accuracy"
        ]
        precision = precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"]
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"]
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"]

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="hyperpartisanship_classification",
        learning_rate=lr,
        lr_scheduler_type="constant",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.001,
        eval_strategy="epoch",
        report_to="wandb",
        logging_steps=5,
        fp16=True,
        gradient_checkpointing=False,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    run_results = trainer.evaluate(tokenized_dataset["test"])
    results.append(run_results)
    print(json.dumps(run_results, indent=4))

avg_results = compute_average_metrics(results)
wandb.log(
    {
        "avg_accuracy": avg_results["eval_accuracy"]["score"],
        "avg_precision": avg_results["eval_precision"]["score"],
        "avg_recall": avg_results["eval_recall"]["score"],
        "avg_f1_score": avg_results["eval_f1-score"]["score"],
    }
)
print(json.dumps(avg_results, indent=4))