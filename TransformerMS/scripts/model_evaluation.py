import os
import torch
import json
import math
import logging
from torch.nn import CrossEntropyLoss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from TransformerMS.Model.model_architecture import GPTLanguageModel
from TransformerMS.Model.model_utils import decode_tokens
from TransformerMS.scripts.tokenizer import load_tokenizer
from bart_score import BARTScorer

# ---------------------- Logging Setup ----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(BASE_DIR, 'Logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(log_dir, 'evaluation.log'))
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# ---------------------- Metric Functions ----------------------

def compute_perplexity(loss):
    return math.exp(loss)

def compute_token_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    return correct.sum().item() / correct.numel()

def compute_bleu(reference_texts, generated_texts):
    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([ref.split()], gen.split(), smoothing_function=smoothie)
        for ref, gen in zip(reference_texts, generated_texts)
    ]
    return sum(scores) / len(scores)

def compute_bart_score(reference_texts, generated_texts):
    scorer = BARTScorer(device='cuda' if torch.cuda.is_available() else 'cpu')
    scores = scorer.score(generated_texts, reference_texts)
    return sum(scores) / len(scores)

# ---------------------- Evaluation Logic ----------------------

def evaluate():
    try:
        # Load tokenizer and model
        tokenizer = load_tokenizer()
        checkpoint_path = os.path.join(BASE_DIR, 'Joblibs', 'gpt_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = GPTLanguageModel(vocab_size=checkpoint['config']['vocab_size'],
                                 block_size=checkpoint['config']['block_size'],
                                 n_embd=checkpoint['config']['n_embd'],
                                 n_head=checkpoint['config']['n_head'],
                                 n_layer=checkpoint['config']['n_layer'],
                                 dropout=checkpoint['config']['dropout'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Sample evaluation data
        samples = [
            ("The future of AI is", "The future of AI is bright."),
            ("The capital of France is", "The capital of France is Paris."),
        ]
        loss_fn = CrossEntropyLoss()

        references = []
        generations = []
        total_loss = 0
        total_acc = 0

        for prompt, true_text in samples:
            idx = tokenizer.encode(prompt)
            x = torch.tensor([idx], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
                acc = compute_token_accuracy(logits, x)
                total_loss += loss.item()
                total_acc += acc

                generated = model.generate(x, max_new_tokens=20)[0].tolist()
                decoded = decode_tokens(generated, tokenizer)
                references.append(true_text)
                generations.append(decoded)

        perplexity = compute_perplexity(total_loss / len(samples))
        accuracy = total_acc / len(samples)
        bleu = compute_bleu(references, generations)
        bart = compute_bart_score(references, generations)

        results = {
            "perplexity": perplexity,
            "accuracy": accuracy,
            "bleu_score": bleu,
            "bart_score": bart,
        }

        # Save results
        metrics_path = os.path.join(log_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Evaluation metrics saved to {metrics_path}")
        logger.info(results)

    except Exception as e:
        logger.error("Evaluation failed", exc_info=True)

if __name__ == "__main__":
    evaluate()
