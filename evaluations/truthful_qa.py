# from datasets import load_metric
from evaluate import load as load_metric

class TruthfulQA_Eval():
    def __init__(self):
        pass
    def evaluate(self, baseline_answer, model_predictions):
        """Evaluate predictions against distorted answers using BLEU and ROUGE scores."""
        print("baseline:", baseline_answer, "\nprediction:",model_predictions)
        baseline_answer = baseline_answer.iloc[:,0].tolist()  # Convert to list of strings
        model_predictions = model_predictions.iloc[:,0].tolist() 

        # Load BLEU and ROUGE metrics
        bleu = load_metric("bleu")
        rouge = load_metric("rouge")
        
        # Prepare references and predictions for BLEU
        references = [[answer] for answer in baseline_answer]
        predictions = [prediction for prediction in model_predictions]
        
        # Compute BLEU score
        bleu_result = bleu.compute(predictions=predictions, references=references)
        
        # Prepare references and predictions for ROUGE (these require raw text, not tokenized)
        references_rouge = [" ".join(ref) for ref in baseline_answer]  # Raw reference answers
        predictions_rouge = [" ".join(pred) for pred in model_predictions]  # Raw model outputs
        
        # Compute ROUGE score
        rouge_result = rouge.compute(predictions=predictions_rouge, references=references_rouge, use_stemmer=True)
        
        # Combine both results in a dictionary and return
        return {
            'bleu': bleu_result['bleu'],
            'rouge1': rouge_result['rouge1'],  # ROUGE-1 score (F1)
            'rouge2': rouge_result['rouge2'],  # ROUGE-2 score (F1)
            'rougeL': rouge_result['rougeL']  # ROUGE-L score (F1)
        }