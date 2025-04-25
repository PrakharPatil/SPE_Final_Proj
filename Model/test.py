# test.py (fixed version)
import torch
import joblib
from model_architecture import GPTLanguageModel
from tokenizers import Tokenizer  # Make sure this is imported


def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = joblib.load('tokenizer.joblib')

    # Add the encode/decode functions to the tokenizer object
    tokenizer.encode_text = lambda text: tokenizer.encode(text).ids
    tokenizer.decode_text = lambda ids: tokenizer.decode(ids)

    # Load model checkpoint
    checkpoint = torch.load('gpt_model.pth', map_location=device)
    config = checkpoint['config']

    # Initialize model
    model = GPTLanguageModel(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer, device


def generate_text(prompt, model, tokenizer, device, max_new_tokens=500):
    # Encode using the tokenizer's method
    encoded = tokenizer.encode(prompt).ids
    context = torch.tensor([encoded], dtype=torch.long, device=device)

    # Generate with proper block size handling
    generated = model.generate(context, max_new_tokens=max_new_tokens)

    # Decode using the tokenizer's method
    return tokenizer.decode(generated[0].tolist())


if __name__ == '__main__':
    model, tokenizer, device = load_model()
    prompt = "Hello "
    print(generate_text(prompt, model, tokenizer, device))