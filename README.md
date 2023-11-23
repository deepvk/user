## Scripts


1. Скрипт для обработки файла с датасетом для задачи классификации (только NLI):
    --filepath          Путь к файлу с датасетом (в формате cvs, json или jsonl)
    --text_columns      Название столбцов с текстом (по-умолчанию sentence)
    --label_column      Столбец с лейблами
    

    `python3 -m scripts.process_nli_dataset --filepath data/rcb/val.jsonl --text_columns "premise,hypothesis" --label_column label` 

2. Скрипт для запуска классификации *run_classification.sh*.


<!-- # py_template

Template repository for Python projects.
Use it to create a new repo, but feel free to adopt for your use-cases.

## Structure

There are several directories to organize your code:
- `src`: Main directory for your modules, e.g., models or dataset implementations, train loops, metrics.
- `scripts`: Directory to define scripts to interact with modules, e.g., run training or evaluation, run data preprocessing, collect statistic.
- `tests`: Directory for tests, this may include multiple unit tests for different parts of logic.

You can create new directories for your need.
For example, you can create a `Notebooks` folder for Jupyter notebooks, such as `EDA.ipynb`.

## Usage

First of all,
navigate to [`pyproject.toml`](./pyproject.toml) and set up `name` and `url` properties according to your project.

For correct work of the import system:
1. Use absolute import statements starting from `src`. For example, `from src.model import MySuperModel`
2. Execute scripts as modules, i.e. use `python -m scripts.<module_name>`. See details about `-m` flag [here](https://docs.python.org/3/using/cmdline.html#cmdoption-m).

To keep your code clean, use `black`, `isort`, and `mypy`
(install everything from [`requirements.dev.txt`](./requirements.dev.txt)).
[`pyproject.toml`](./pyproject.toml) already defines their parameters, but you can change them if you want.
 -->