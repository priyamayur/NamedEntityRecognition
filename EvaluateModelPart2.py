from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import AdamW, AutoModelForSequenceClassification
import evaluate
import numpy as np
import torch
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
    masked_test_data = load_dataset("csv",data_files="data/experiment2_test_data.csv",sep='\t', index_col=0);
    masked_valid_data = load_dataset("csv",data_files="data/experiment2_valid_data.csv",sep='\t', index_col=0);
    masked_training_data = load_dataset("csv",data_files="data/experiment2_training_data.csv",sep='\t', index_col=0);

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
#metric = evaluate.combine(["f1", "precision", "recall"])
#metric = evaluate.load('accuracy')


def compute_metrics_accuracy(eval_pred):
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics_f1(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

def compute_metrics_precision(eval_pred):
    metric = evaluate.load('precision')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

def compute_metrics_recall(eval_pred):
    metric = evaluate.load('recall')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')


def evaluate_model(test_set, model_name= 'NeighbourNER_model', metric_type = 'accuracy'):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, id2label={0: 'non-NER', 1: 'B', 2: 'I'} )
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    if metric_type == 'accuracy':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_set,
            compute_metrics=compute_metrics_accuracy,
        )
    elif metric_type =='f1':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_set,
            compute_metrics=compute_metrics_f1,
        )
    elif metric_type =='precision':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_set,
            compute_metrics=compute_metrics_precision,
        )
    elif metric_type =='recall':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_set,
            compute_metrics=compute_metrics_recall,
        )

    eval = trainer.evaluate()
    return eval





print("3-class classification")
print("test dataset 1")
eval_valid = evaluate_model(valid_dataset, 'NeighbourNER_experiment2_valid', 'accuracy')
print(eval_valid)
eval_valid = evaluate_model(valid_dataset, 'NeighbourNER_experiment2_valid', 'f1')
print(eval_valid)
eval_valid = evaluate_model(valid_dataset, 'NeighbourNER_experiment2_valid', 'precision')
print(eval_valid)
eval_valid = evaluate_model(valid_dataset, 'NeighbourNER_experiment2_valid', 'recall')
print(eval_valid)
print("test dataset 2")
eval_test = evaluate_model(test_dataset, 'NeighbourNER_experiment2_valid', 'accuracy')
print(eval_test)
eval_test = evaluate_model(test_dataset, 'NeighbourNER_experiment2_valid', 'f1')
print(eval_test)
eval_test = evaluate_model(test_dataset, 'NeighbourNER_experiment2_valid', 'precision')
print(eval_test)
eval_test = evaluate_model(test_dataset, 'NeighbourNER_experiment2_valid', 'recall')
print(eval_test)
