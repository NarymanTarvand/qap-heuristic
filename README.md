# qap-heuristic

The quadratic assignment problem (QAP) is one of the most difficult problems in the NP-hard class. This repository explores the following heuristics to solve a QAP,

- constructive heuristic(s)
- local search heuristic(s)
- metaheuristic(s)

## Getting set up

```
pip install -r requirements.txt
```

## Running the algorithms

### GRASP

```
python -m metaheuristics.GRASP.randomised_greedy_grasp <instance_name> <restricted_time> <search_method>
```

for example, `python -m metaheuristics.GRASP.randomised_greedy_grasp bur26a 0 "local search"`

### Genetic Algorithm

```
python -m metaheuristics.genetic_algorithm <instance_name> <restricted_time>
```

for example, `python -m metaheuristics.genetic_algorithm bur26a 0`

### Local Search

```
python -m local_search_heuristics.local_search <instance_name> <restricted_time>
```

### Multistart Local Search

```
python -m local_search_heuristics.local_search <instance_name> <restricted_time> <n_multistart>
```

### Running all algorithms in parallel

```
python -m heuristic_performance_test <restricted_time> <subset>
```

The subset value can be used to run the first n instances. If `subset` is set as `0` the algorithm will run for all instances, otherwise if `subset` is set to an integer value `x` it will run for the first `x` instances.

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
