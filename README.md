# Frequency Supported Neural Networks

## Paper

Preprint is available on [https://arxiv.org/abs/2305.06344](https://arxiv.org/abs/2305.06344) and in `papers` directory.

### Abstract

Neural networks are a very general type of model capable of learning various relationships between multiple variables. One example of such relationships, particularly interesting in practice, is the input-output relation of nonlinear systems, which has a multitude of applications. Studying models capable of estimating such relation is a broad discipline with numerous theoretical and practical results. Neural networks are very general, but multiple special cases exist, including convolutional neural networks and recurrent neural networks, which are adjusted for specific applications, which are image and sequence processing respectively. We formulate a hypothesis that adjusting general network structure by incorporating frequency information into it should result in a network specifically well suited to nonlinear system identification. Moreover, we show that it is possible to add this frequency information without the loss of generality from a theoretical perspective. We call this new structure Frequency-Supported Neural Network (FSNN) and empirically investigate its properties.

## Code 

Code used to develop and run experiments in transferred from core repository, [dynamical-neural-networks](https://github.com/kzajac97/dynamical-neural-networks/tree/main).
This repository contains unmaintained code state used to run experiments for the paper and allowing to reproduce them.

## Data

Three datasets were used in the paper:
* Static Affine Benchmark
* Wiener-Hammerstein Benchmark
* Silverbox Benchmark

Static Affine Benchmark can be generated using provided notebook (`notebooks/static-affine-benchmark-generation.ipynb`) 
or accessed directly from data directory (`data/static-affine-benchmark.csv`).

Wiener Hammerstein Benchmark and Silverbox Benchmark
(along with many others) can be accessed from [https://www.nonlinearbenchmark.org/](www.nonlinearbenchmark.org/).

## Experiments

Experiments were run using code provided in `src` directory and notebook `notebooks/experiment.ipynb`, which can be
configured for multiple benchmarks and models. Detailed description is provided in the notebook. To run experiment, 
WANDB account is required, since it was used as experiment management tool. Numerical results of experiments
are provided in the data directory, one file for each benchmark.

WANDB projects are available under following links:
* [https://wandb.ai/kzajac/fsnn-static-affine](https://wandb.ai/kzajac/fsnn-static-affine)
* [https://wandb.ai/kzajac/fsnn-wiener-hammerstein/overview](https://wandb.ai/kzajac/fsnn-wiener-hammerstein/overview)
* [https://wandb.ai/kzajac/fsnn-silverbox](https://wandb.ai/kzajac/fsnn-silverbox)

## Requirements

Listed in `requirements.txt` file, to run using Docker see: [Dockerfile @ dynamical-neural-networks]([https://github.com/kzajac97/dynamical-neural-networks/tree/main](https://github.com/kzajac97/dynamical-neural-networks/blob/main/Dockerfile)). 
