# qap-heuristic

The quadratic assignment problem (QAP) is one of the most difficult problems in the NP-hard class. This repository explores the following heuristics to solve a QAP,

- constructive heuristic(s)
- local search heuristic(s)
- metaheuristic(s)

## Getting set up

```
pip install -r requirements.txt
```

### Running code quality checks

This project expects committed code to be compliant with the following code quality tools:

| Name                                      | Description                |
| ----------------------------------------- | -------------------------- |
| [flake8](https://github.com/PyCQA/flake8) | Linting                    |
| [black](https://github.com/python/black)  | Opinionated auto-formatter |
| [mypy](https://github.com/python/mypy)    | Static type checking       |

There are a couple of ways to run these checkers over your local codebase.

#### Using command line

```bash
<flake8|black|mypy> qap-heuristic/
```

#### Using pre-commit hooks

This project provides configuration for _optional_ pre-commit hooks using [pre-commit](https://github.com/pre-commit/pre-commit).

To install and create the pre-commit hooks, with your [local virtualenv active](#local-python-environment), run:

```bash
pre-commit install --install-hooks
```
