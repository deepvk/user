from typing import List, Optional

import click
import pandas as pd

tokenizer = {
    "entailment": 0,
    "логическое следование": 0,
    "neutral": 1,
    "нейтральная взаимосвязь": 1,
    "not_entailment": 1,
    "contradiction": 2,
    "противоречие": 2,
}

"""
#TBD write comment
"""


def process_nli_dataset(
    filepath: str,
    text_columns: Optional[str] = None,
    label_column: Optional[str] = None,
    save_path: Optional[str] = None,
):
    ext = filepath.split(".")[-1]
    if ext not in ["csv", "json", "jsonl"]:
        print("Extension shoud be one of csv, json, jsonl.")
    else:
        if ext == "csv":
            data = pd.read_csv(filepath)
        elif ext == "json":
            data = pd.read_json(filepath)
        else:
            data = pd.read_json(filepath, lines=True)

        if text_columns is None:
            text_columns = "sentence"

        if label_column is None:
            label_column = "label"

        if save_path is None:
            save_path = "".join(filepath.split(".")[:-1]) + "_processed.csv"

        # Concatenate all text_columns to one with ',' as a delimeter
        text_column_names = text_columns.split(",")
        data["sentence"] = data[text_column_names[0]]
        for column in text_column_names[1:]:
            data["sentence"] += "," + data[column]

        # Project str labels to unified int labels
        labels = data[label_column]
        data["label"] = [tokenizer[label.lower()] for label in labels]

        # Leave only valuable columns for out task
        data = data[["sentence", "label"]]
        data.to_csv(save_path)


@click.command()
@click.option("--filepath", type=str)
@click.option("--text_columns", type=str, default="sentence")
@click.option("--label_column", type=str, default="label")
@click.option("--save_path", type=str, default=None)
def main(filepath, text_columns, label_column, save_path):
    process_nli_dataset(filepath, text_columns, label_column, save_path)


if __name__ == "__main__":
    main()
