import os
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

def preprocess_function(examples):
    """
    Preprocesses the input examples for T5.

    Args:
        examples (dict): A dictionary containing the input examples with 'incorrect' and 'correct' keys.

    Returns:
        dict: A dictionary containing the preprocessed model inputs with 'input_ids' and 'labels' keys.
    """
    # Process each example in the batch
    model_inputs = tokenizer(["correct: " + example for example in examples['incorrect']], padding="max_length", truncation=True, max_length=128)
    
    # Prepare labels for each example in the batch
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['correct'], padding="max_length", truncation=True, max_length=128)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



### Data Preparation ###

# Base directory where the dataset files are located
base_path = "datasets\\CDISC-SDTM sample study harvard"

# Filenames for the faulty data
filenames_faulty = [
    "dm-faulty-1.csv",
    "dm-faulty-2.csv",
    "dm-faulty-3.csv",
    "dm-faulty-4.csv",
    "dm-faulty-5.csv"
]

# Filenames for the correct data
filename_correct = "dm-correct.csv"
filenames_correct = [filename_correct] * 5

# Initialize a list to hold DataFrame objects
merged_dataframes = []

# Loop through each pair of faulty and correct filenames
for filename_faulty, filename_correct in zip(filenames_faulty, filenames_correct):
    # Construct the full file paths for the faulty and correct data files
    path_faulty = os.path.join(base_path, filename_faulty)
    path_correct = os.path.join(base_path, filename_correct)
    
    # Load the faulty data, selecting only the 'SEX' column and renaming it to 'incorrect'
    df_faulty_tmp = pd.read_csv(path_faulty, usecols=["SEX"]).rename(columns={"SEX": "incorrect"})
    # Load the correct data in a similar manner, renaming the 'SEX' column to 'correct'
    df_correct_tmp = pd.read_csv(path_correct, usecols=["SEX"]).rename(columns={"SEX": "correct"})
    
    # Merge the faulty and correct DataFrames
    df_merged_tmp = pd.merge(df_faulty_tmp, df_correct_tmp, how="inner", left_index=True, right_index=True)
    # Add the merged DataFrame to the list for later concatenation
    merged_dataframes.append(df_merged_tmp)

# Concatenate all the merged DataFrames into a single DataFrame
df = pd.concat(merged_dataframes, ignore_index=True)

# Final DataFrame
df

# # Alternative dataset
# data = [
#     {'incorrect': 'male', 'correct': 'M'},
#     {'incorrect': 'M', 'correct': 'M'},
#     {'incorrect': 'Mle', 'correct': 'M'},
#     {'incorrect': 'Mal', 'correct': 'M'},
#     {'incorrect': 'mael', 'correct': 'M'},
#     {'incorrect': 'female', 'correct': 'F'},
#     {'incorrect': 'F', 'correct': 'F'},
#     {'incorrect': 'Femle', 'correct': 'F'},
#     {'incorrect': 'Fmale', 'correct': 'F'},
#     {'incorrect': 'femal', 'correct': 'F'}
# ]
# df = pd.DataFrame(data)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)


### Training ###

# Load the tokenizer and model
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Check if CUDA is available
if torch.cuda.is_available():
    print("Training on GPU.")
    model = model.cuda()  # Move the model to GPU
else:
    print("Training on CPU.")
    
# Preprocess the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Arguments for the Trainer
training_args = TrainingArguments(
    output_dir=".\\results\\results-t5", 
    evaluation_strategy="epoch", 
    learning_rate=3e-4, 
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    num_train_epochs=100, 
    weight_decay=0.01,  # Weight decay for regularization
    warmup_steps=500,  # Number of warmup steps for the learning rate scheduler
    logging_dir='.\\results\\logs-t5', 
    logging_steps=50, 
    max_grad_norm=1.0,  # Maximum gradient norm for gradient clipping
    save_strategy="epoch",  # Save model checkpoint at the end of each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Metric used to identify the best model
    greater_is_better=False  # Lower eval_loss indicates a better model
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
    )

trainer.train()

# tensorboard --logdir=./results/logs-t5
