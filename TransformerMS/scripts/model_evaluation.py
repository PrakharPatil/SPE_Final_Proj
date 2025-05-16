# scripts/model_evaluation.py
import os
import torch
import json
import math
import logging
from torch.nn import CrossEntropyLoss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dvclive import Live

from TransformerMS.Model.model_architecture import GPTLanguageModel
from TransformerMS.Model.model_utils import decode_tokens
# from TransformerMS.scripts.tokenizer import load_tokenizer
# from bart_score import BARTScorer

# ---------------------- Configuration ----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'gpt_model.pth')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'Joblibs', 'tokenizer.joblib')
LOG_DIR = os.path.join(BASE_DIR, 'Logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------- Logging Setup ----------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'evaluation.log'))
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)


# ---------------------- Evaluation Class ----------------------
class ModelEvaluator:
    def __init__(self, params):
        self.params = params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load components
        # self.tokenizer = load_tokenizer(TOKENIZER_PATH)
        self.model = self._load_model()
        self.loss_fn = CrossEntropyLoss()
        # self.bart_scorer = BARTScorer(device=self.device)

    def _load_model(self):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        model = GPTLanguageModel(
            vocab_size=checkpoint['config']['vocab_size'],
            block_size=checkpoint['config']['block_size'],
            n_embd=checkpoint['config']['n_embd'],
            n_head=checkpoint['config']['n_head'],
            n_layer=checkpoint['config']['n_layer'],
            dropout=checkpoint['config']['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model.to(self.device)

    def _compute_metrics(self, references, generations, total_loss):
        return {
            "perplexity": math.exp(total_loss / len(references)),
            "bleu_score": self._calculate_bleu(references, generations),
            "bart_score": self._calculate_bart(references, generations),
        }

    def _calculate_bleu(self, references, generations):
        smoothie = SmoothingFunction().method4
        scores = [
            sentence_bleu([ref.split()], gen.split(), smoothing_function=smoothie)
            for ref, gen in zip(references, generations)
        ]
        return sum(scores) / len(scores)

    def _calculate_bart(self, references, generations):
        scores = self.bart_scorer.score(generations, references)
        return sum(scores) / len(scores)

    def evaluate_samples(self, samples):
        references = []
        generations = []
        total_loss = 0

        for prompt, true_text in samples:
            # Encode input
            idx = self.tokenizer.encode(prompt)
            x = torch.tensor([idx], dtype=torch.long).to(self.device)

            # Generate prediction
            with torch.no_grad():
                logits, _ = self.model(x)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
                total_loss += loss.item()

                generated = self.model.generate(
                    x,
                    max_new_tokens=self.params['evaluation']['max_new_tokens']
                )[0].tolist()

                decoded = decode_tokens(generated, self.tokenizer)
                generations.append(decoded)
                references.append(true_text)

        return self._compute_metrics(references, generations, total_loss)


# ---------------------- Main Execution ----------------------
def main():
    with Live(save_dvc_exp=True) as live:
        try:
            # Load parameters
            params_path = os.path.join(BASE_DIR, 'params.yaml')
            with open(params_path) as f:
                params = yaml.safe_load(f)

            # Initialize and evaluate
            evaluator = ModelEvaluator(params)
            results = evaluator.evaluate_samples(params['evaluation']['samples'])

            # Log metrics
            for metric, value in results.items():
                live.log_metric(metric, value)
                logger.info(f"{metric}: {value:.4f}")

            # Save results
            metrics_path = os.path.join(BASE_DIR, 'reports', 'metrics.json')
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=4)

            logger.info(f"Metrics saved to {metrics_path}")
            return 0

        except Exception as e:
            logger.error("Evaluation failed", exc_info=True)
            live.log_metric("error", 1)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())