Для сборки и установки библиотек:

1. `python3 enkodechka/setup.py build`
2. `python3 enkodechka/setup.py install`


Для запуска валидации:

`python3 validate.py --model_path <path to model> --instr`

Параметры:
- `--model_path` - путь до папки с моделью.
- `--instr` - валидация с инструкциями. 



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
