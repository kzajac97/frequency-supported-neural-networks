# Frequency Supported Neural Networks

## Paper

Preprint is available on [arXiv](https://arxiv.org/abs/2305.06344) and in `papers` directory.

## Code 

Code used to develop and run experiments in transferred from core repository, [dynamical-neural-networks](https://github.com/kzajac97/dynamical-neural-networks/tree/main).
This repository contains unmaintained code state used to run experiments for the paper.

## Data

Three datasets were used in the paper:
* Static Affine Benchmark
* Wiener Hammerstein Benchmark
* Silverbox Benchmark

Static Affine Benchmark can be generated using provided notebook (`notebooks/static-affine-benchmark-generation.ipynb`) 
or accessed directly from data directory (`data/static-affine-benchmark.csv`). Wiener Hammerstein Benchmark and Silverbox Benchmark
(along with many others) can be accessed from [https://www.nonlinearbenchmark.org/](www.nonlinearbenchmark.org/).

## Experiments

Experiments were run using code provided in `src` directory and notebook `notebooks/experiment.ipynb`, which can be
configured for multiple benchmarks and models. Detailed description is provided in the notebook. To run experiment, 
WANDB account is required, since it was used as experiment management tool. Numerical results of experiments
are provided in the data directory, one file for each benchmark.

*Note*: WANBD report with logs of all runs and plots will be published.