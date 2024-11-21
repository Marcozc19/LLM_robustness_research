import pandas as pd

def main():
    df = pd.read_json("LLM_robustness_research/data/truthful_qa/truthful_qa.json")
    df_question = df[["question"]]
    df_question.columns = ["question"]
    return df_question