import sys
from argparse import ArgumentParser
from datetime import datetime

import run_classification
from datasets import Value, load_dataset
from run_classification import DataTrainingArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_files = {"train": data_args.train_file}
    if data_args.train_file.endswith(".csv"):
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    datetime_now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    training_args.output_dir = "tmp/ru_wanli" + datetime_now + "/"
    training_args.logging_dir = "tmp/ru_wanli" + datetime_now + "/"
    training_args.run_name = training_args.logging_dir

    logging_steps = len(raw_datasets["train"]) // training_args.per_device_train_batch_size // 3
    training_args.logging_steps = logging_steps
    training_args.save_steps = logging_steps
    training_args.eval_steps = logging_steps

    run_classification.main((model_args, data_args, training_args))


if __name__ == "__main__":
    main()
