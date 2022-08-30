# Mlflow - MLProject

This MLproject aims to chain together multiple different MLflow runs which each encapsulate training step, allowing caching and reusing intermediate results.

There are two steps to this workflow:

- `train.py`: Trains the provided model and saves it in ``results`` folder.

- `validate.py`: Loads the trained model from train.py and validates the data by calculating the accuracy score.

While we can run each of these steps manually, here we have a driver run, defined as **main** (main.py). This run will run the steps in order passing the results of one to the next. Additionally, this run will attempt to determine if a sub-run has already been executed successfully with the same parameters and, if so, reuse the cached results.

## Installation

- Install MLflow from PyPI via ``pip install mlflow``
- MLflow requires ``conda`` to be on the ``PATH`` for the projects feature.

## Run the project

In order for this workflow, you must execute ``mlflow run`` from this directory. This will create a ``mlruns`` directory where all the things will be stored. You can simply run:

```
mlflow run .
```

You can also try defining the parameter if required in main.py to pass the parameters:

```
mlflow run . -P paramter_name=parameter
```
