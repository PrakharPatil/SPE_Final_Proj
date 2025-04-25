import torch
import joblib
from .model_architecture import GPTLanguageModel
# from tokenizers import Tokenizer

class ModelLoader:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer = self._load_model()
        self.model.eval()

    def _load_model(self):
        # Load tokenizer
        tokenizer = joblib.load('Joblibs/tokenizer.joblib')

        # Add the encode/decode functions to the tokenizer object
        tokenizer.encode_text = lambda text: tokenizer.encode(text).ids
        tokenizer.decode_text = lambda ids: tokenizer.decode(ids)

        # Load model checkpoint
        checkpoint = torch.load('Joblibs/gpt_model.pth', map_location=self.device)
        config = checkpoint['config']

        # Initialize model
        model = GPTLanguageModel(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            dropout=config['dropout']
        ).to(self.device)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, tokenizer

    def generate_response(self, prompt: str, max_new_tokens: int = 30):
        # Encode prompt
        encoded = self.tokenizer.encode(prompt).ids
        context = torch.tensor([encoded], dtype=torch.long, device=self.device)

        # Generate text
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens=max_new_tokens)

        # Decode and return
        return self.tokenizer.decode(generated[0].tolist())