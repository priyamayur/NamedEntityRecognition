import pandas as pd
from transformers import AutoTokenizer
from transformers import DistilBertTokenizerFast
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
from accelerate import Accelerator
import tqdm
from sklearn.metrics import confusion_matrix
from datasets import load_dataset, load_metric

raw_datasets = load_dataset("imdb")



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
    # masked_training_valid_data = pd.read_pickle("data/masked_training_valid_data")
    # masked_test_data = pd.read_pickle("data/masked_test_data")
    # masked_valid_data = pd.read_pickle("data/masked_valid_data")
    # masked_training_data = pd.read_pickle("data/masked_training_data")

    # masked_test_data = pd.read_pickle("data/masked_small_test_data")
    # masked_valid_data = pd.read_pickle("data/masked_small_valid_data")
    # masked_training_data = pd.read_pickle("data/masked_small_training_data")

    # masked_test_data = load_dataset("csv",data_files="data/masked_small_test_data.csv",sep='\t', index_col=0);
    # masked_valid_data = load_dataset("csv",data_files="data/masked_small_valid_data.csv",sep='\t', index_col=0);
    # masked_training_data = load_dataset("csv",data_files="data/masked_small_training_data.csv",sep='\t', index_col=0);

    masked_test_data = load_dataset("csv",data_files="data/experiment3_test_data.csv",sep='\t', index_col=0);
    masked_valid_data = load_dataset("csv",data_files="data/experiment3_valid_data.csv",sep='\t', index_col=0);
    masked_training_data = load_dataset("csv",data_files="data/experiment3_training_data.csv",sep='\t', index_col=0);

    return masked_training_data, masked_valid_data, masked_test_data


training_data, valid_data, test_data = load_data()


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# print(small_train_dataset)


tokenized_train = training_data.map(tokenize_function, batched=True)
train_dataset = tokenized_train["train"].shuffle(seed=42)
print(train_dataset)

tokenized_test = test_data.map(tokenize_function, batched=True)
test_dataset = tokenized_test["train"].shuffle(seed=42)
print(test_dataset)

tokenized_valid = valid_data.map(tokenize_function, batched=True)
valid_dataset = tokenized_valid["train"].shuffle(seed=42)
print(valid_dataset)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
training_args = TrainingArguments(output_dir="test_trainer")
#metric = evaluate.combine(["accuracy", "precision", "recall", "f1"])
# metric = evaluate.combine([
#     evaluate.load("accuracy", average="weighted"),
#     evaluate.load("precision", average="weighted"),
#     evaluate.load("recall", average="weighted"),
#     evaluate.load("f1", average="weighted")
# ])

metric = evaluate.load('f1', average='weighted')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    #return metric.compute(predictions=predictions, references=labels)
    return metric.compute(predictions=predictions, references=labels,  average='weighted')


def train_model(test_set, model_name= 'NeighbourNER_model'):

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(model_name)
    return model

print("experiment 3 NeighbourNER_model_valid")
model_v = train_model(valid_dataset, "NeighbourNER_experiment3_valid")


