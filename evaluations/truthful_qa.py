# from datasets import load_metric
from evaluate import load as load_metric
import pandas as pd

class TruthfulQA_Eval():
    def __init__(self):
        pass
    def evaluate(self, baseline_answer,  pred_csv):
        """Evaluate predictions against baseline answers using BLEU and ROUGE scores per example,
        save the results to a CSV file, and compute average scores."""
        # Convert DataFrames to lists of strings
        model_predictions = pred_csv['response']
        baseline_answer = [" ".join(ref) if isinstance(ref, list) else ref for ref in baseline_answer]
        model_predictions = [" ".join(pred) if isinstance(pred, list) else pred for pred in model_predictions]
        
        perplexity = pred_csv['perplexity']
        avg_perplexity = perplexity.mean()

        # Load metrics
        bleu_metric = load_metric("bleu")
        rouge_metric = load_metric("rouge")
        
        # Prepare lists to store per-example data
        references = []
        predictions = []
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # Iterate over each prediction and reference
        for pred, ref in zip(model_predictions, baseline_answer):
            # Append reference and prediction
            references.append(ref)
            predictions.append(pred)
            
            # Compute BLEU score per example with complete strings
            bleu_result = bleu_metric.compute(predictions=[pred], references=[[ref]])  # Use full strings
            bleu_score = bleu_result['bleu']
            bleu_scores.append(bleu_score)
            
            # Compute ROUGE scores per example
            rouge_result = rouge_metric.compute(predictions=[pred], references=[ref], use_stemmer=True)
            rouge1 = rouge_result['rouge1']
            rouge2 = rouge_result['rouge2']
            rougeL = rouge_result['rougeL']
            
            rouge1_scores.append(rouge1)
            rouge2_scores.append(rouge2)
            rougeL_scores.append(rougeL)
        
        # Create a DataFrame with the collected data
        results_df = pd.DataFrame({
            'reference': references,
            'prediction': predictions,
            'bleu': bleu_scores,
            'rouge1': rouge1_scores,
            'rouge2': rouge2_scores,
            'rougeL': rougeL_scores,
            'perplexity': perplexity
        })
        
        # Compute average scores
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        
        print("\nAverage Scores:")
        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2 F1: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
        print(f"Average Perplexity: {avg_perplexity:.4f}")
        
        # Return the DataFrame and average scores if needed
        return results_df, {
            'avg_bleu': avg_bleu,
            'avg_rouge1': avg_rouge1,
            'avg_rouge2': avg_rouge2,
            'avg_rougeL': avg_rougeL,
            'avg_perplexity': avg_perplexity
        }