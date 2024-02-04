import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric
from torch.utils.data import Dataset
import torch
import json

# Base directory
base_path = "datasets/CDISC-SDTM sample study harvard"

# Filenames for train and valid datasets
filenames_train = ["ae-1.csv", "dm-1.csv", "eg-1.csv", "lb-1.csv", "mh-1.csv", "pe-1.csv", "vs-1.csv"]
filenames_valid = ["ae-2.csv", "dm-2.csv", "eg-2.csv", "lb-2.csv", "mh-2.csv", "pe-2.csv", "vs-2.csv"]

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ColumnDataDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_token_len=512):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item_index):
        text = self.inputs[item_index]
        label = self.labels[item_index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to process files and extract data
def process_files(filenames, base_path):
    inputs = []
    labels = []
    unique_column_names = set()

    for filename in filenames:
        file_path = f"{base_path}/{filename}"
        df = pd.read_csv(file_path)
        unique_column_names.update(df.columns)

        for column in df.columns:
            aggregated_text = ' '.join([str(x) for x in df[column].dropna().values])
            inputs.append(aggregated_text)
            labels.append(column_name_to_label[column])  # Directly use the mapping

    return inputs, labels, unique_column_names

# Function to collect unique column names from files
def collect_unique_column_names(filenames, base_path):
    unique_column_names = set()
    for filename in filenames:
        file_path = f"{base_path}/{filename}"
        df = pd.read_csv(file_path)
        unique_column_names.update(df.columns)
    return unique_column_names

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Collect unique column names from both training and validation datasets
unique_column_names_train = collect_unique_column_names(filenames_train, base_path)
unique_column_names_valid = collect_unique_column_names(filenames_valid, base_path)
all_unique_column_names = unique_column_names_train.union(unique_column_names_valid)

# Create a mapping from column names to numeric labels based on all unique column names
column_name_to_label = {column_name: i for i, column_name in enumerate(all_unique_column_names)}

column_name_to_label

# Save the column_name_to_label mapping to a file 
column_name_to_label_path = "config/column_name_to_label.json"
with open(column_name_to_label_path, 'w') as file:
    json.dump(column_name_to_label, file)
    
# Process training and validation files
train_inputs, train_labels, _ = process_files(filenames_train, base_path)
eval_inputs, eval_labels, _ = process_files(filenames_valid, base_path)

# Create datasets
train_dataset = ColumnDataDataset(train_inputs, train_labels, tokenizer)
eval_dataset = ColumnDataDataset(eval_inputs, eval_labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results/results-bert",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./results/logs-bert",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=3e-5,
)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(column_name_to_label))

# Load metric
accuracy_metric = load_metric("accuracy")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# tensorboard --logdir=./results/logs-bert