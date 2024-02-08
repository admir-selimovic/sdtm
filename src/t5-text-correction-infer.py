from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def correct_text(text, model, tokenizer):
    """
    Corrects the given text using T5.

    Args:
        text (str): The text to be corrected.
        model: The T5 model used for text correction.
        tokenizer: The tokenizer used for encoding and decoding text.

    Returns:
        str: The corrected text.
    """
    # Prepend "correct: " to the input text
    input_text = "correct: " + text

    # Encode the input text using the tokenizer
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    # Adjust generation settings: increase the number of beams, adjust early stopping, etc.
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    
    # Decode the generated output and remove special tokens
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return corrected_text


# Load the tokenizer and model
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained("./models/t5")

# Check if CUDA is available
if torch.cuda.is_available():
    print("Model on GPU.")
    model = model.cuda()  # Move the model to GPU
else:
    print("Model on CPU.")

print(correct_text("mal", model, tokenizer))
