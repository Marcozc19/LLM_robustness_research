import pandas as pd

def main():
    df = pd.read_json("data/truthful_qa/truthful_qa.json")
    df_question = df[["question"]]
    return df_question