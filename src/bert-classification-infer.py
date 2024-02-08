import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

def process_file_for_inference(filename: str, base_path: str, tokenizer, max_token_len: int = 512):
    """
    Process a file for inference using BERT classification.

    Args:
        filename (str): The name of the file to be processed.
        base_path (str): The base path where the file is located.
        tokenizer: The tokenizer object used for encoding the text.
        max_token_len (int, optional): The maximum length of tokens. Defaults to 512.

    Returns:
        list: A list of dictionaries containing the input IDs and attention masks.

    """
    file_path = f"{base_path}/{filename}"
    df = pd.read_csv(file_path)

    inputs = []
    for column in df.columns:
        aggregated_text = ' '.join([str(x) for x in df[column].dropna().values])
        encoding = tokenizer.encode_plus(
            aggregated_text,
            add_special_tokens=True,
            max_length=max_token_len,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        inputs.append({
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        })

    return inputs

def predict_column_names_with_probs(inputs: list, model: BertForSequenceClassification, column_name_to_label: dict):
    """
    Predicts column names with their corresponding probabilities for a given input using a trained model.

    Args:
        inputs (list): A list of input dictionaries containing 'input_ids' and 'attention_mask'.
        model: The trained model for prediction.
        column_name_to_label (dict): A dictionary mapping column names to label indices.

    Returns:
        predictions (list): A list of predicted column names.
        confidence_scores (list): A list of confidence scores for each prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode
    predictions = []
    confidence_scores = []

    for input in inputs:
        input_ids = input['input_ids'].to(device).unsqueeze(0)  # Add batch dimension and move to device
        attention_mask = input['attention_mask'].to(device).unsqueeze(0)  # Add batch dimension and move to device

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_label_idx = torch.argmax(probs, dim=1).item()
            confidence_score = probs.max().item()

            # Reverse mapping to get column name from label
            label_to_column_name = {v: k for k, v in column_name_to_label.items()}
            predicted_label = label_to_column_name[predicted_label_idx]
            predictions.append(predicted_label)
            confidence_scores.append(confidence_score)

    return predictions, confidence_scores


# Define the path to load the trained model
model_path = "models/sbert"


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path)

# Check if CUDA is available
if torch.cuda.is_available():
    print("Model on GPU.")
    model = model.cuda()  # Move the model to GPU
else:
    print("Model on CPU.")

# Load the column name mapping from a JSON file
column_name_to_label_path = "config/column_name_to_label.json"
with open(column_name_to_label_path, 'r') as file:
    column_name_to_label = json.load(file)

# Reverse mapping from label to column name
label_to_column_name = {v: k for k, v in column_name_to_label.items()}


# Inference inputs
base_path = "datasets/CDISC-SDTM sample study harvard"
filename = "lb-2-bert-class-test.csv"

# Process the file for inference
inference_inputs = process_file_for_inference(filename, base_path, tokenizer)

# Inference function call
predicted_column_names, confidence_scores = predict_column_names_with_probs(inference_inputs, model, column_name_to_label)

# Read the new data from a CSV file
new_data_df = pd.read_csv(f"{base_path}/{filename}")

# Get the actual column names as the 'true' labels for validation
actual_column_names = list(new_data_df.columns)

# Actual column names as the 'true' labels for validation
correct_predictions = sum(1 for actual, predicted in zip(actual_column_names, predicted_column_names) if actual == predicted)
total_predictions = len(predicted_column_names)
accuracy = correct_predictions / total_predictions

# Print the results
for actual, predicted, confidence in zip(actual_column_names, predicted_column_names, confidence_scores):
    print(f"Actual: {actual}, Predicted: {predicted}, Confidence: {confidence:.2f}")

# Save the DataFrame to a CSV file
df = pd.DataFrame({
    'Actual': actual_column_names,
    'Predicted': predicted_column_names,
    'Confidence': confidence_scores
    })
output_file_path = 'predictions_with_confidence.csv' 
df.to_csv(output_file_path, index=False) 

# Calculate the number of correct and incorrect predictions
correct_predictions = sum(1 for actual, predicted in zip(actual_column_names, predicted_column_names) if actual == predicted)
incorrect_predictions = total_predictions - correct_predictions

# Calculate the percentage of correct predictions
percentage_correct = (correct_predictions / total_predictions) * 100

# Print the results
print(f"\nNumber of correct predictions: {correct_predictions}")
print(f"Number of incorrect predictions: {incorrect_predictions}")
print(f"Percentage of correct predictions: {percentage_correct:.2f}%")

