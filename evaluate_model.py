from transformers import TrainingArguments, Trainer
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler,TextClassificationPipeline
from transformers import pipeline
import pandas as pd
import os
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data():
    masked_test_data = load_dataset("csv",data_files="data/masked_small_test_data.csv",sep='\t', index_col=0);
    masked_valid_data = load_dataset("csv",data_files="data/masked_small_valid_data.csv",sep='\t', index_col=0);
    masked_training_data = load_dataset("csv",data_files="data/masked_small_training_data.csv",sep='\t', index_col=0);

    # masked_test_data = load_dataset("csv",data_files="data/experiment2_test_data.csv",sep='\t', index_col=0);
    # masked_valid_data = load_dataset("csv",data_files="data/experiment2_valid_data.csv",sep='\t', index_col=0);
    # masked_training_data = load_dataset("csv",data_files="data/experiment2_training_data.csv",sep='\t', index_col=0);

    return masked_training_data, masked_valid_data, masked_test_data

training_data, valid_data, test_data = load_data()


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train = training_data.map(tokenize_function, batched=True)
train_dataset = tokenized_train["train"].shuffle(seed=42).select(range(5))

tokenized_test = test_data.map(tokenize_function, batched=True)
test_dataset = tokenized_test["train"].shuffle(seed=42).select(range(5))

tokenized_valid = valid_data.map(tokenize_function, batched=True)
valid_dataset = tokenized_valid["train"].shuffle(seed=42).select(range(5))


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def evaluate_model(test_set, model_name= 'NeighbourNER_model'):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, id2label={0: 'non-NER', 1: 'NER'} ) # modify labels as needed.

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
    )
    eval = trainer.evaluate()
    return eval


eval_valid = evaluate_model(valid_dataset, 'NeighbourNER_model_valid')
eval_test = evaluate_model(test_dataset, 'NeighbourNER_model_valid')

print("binary classification")
print("test dataset 1")
print(eval_valid)
print("test dataset 2")
print(eval_test)
