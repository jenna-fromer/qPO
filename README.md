# Batched Bayesian optimization with correlated candidate uncertainties

This folder contains the code to reproduce the results in the manuscript entitled "Batched Bayesian optimization with correlated candidate uncertainties". 

## Installation 

```bash
conda create -n qpo python=3.12
conda activate qpo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt 
```

You can modify the third line with your preferences according to the directions [here](https://pytorch.org/get-started/locally/). 

## Running experiments 

All experiments are contained in the [scripts folder](scripts/). 

## Organization 

qPO (multipoint probability of optimality) and baseline acquisition functions can be found [here](acquisition_functions.py). The iterative Bayesian optimization loop is implemented in [run.py](run.py) and explores the [included datasets](data/), QM9<sup>1,2</sup> and an experimental screen for antibiotics<sup>3</sup>. We use a Tanimoto Gaussian Process as the surrogate model. Its implementation, adapted from the [Practical Molecular Optimization benchmark](https://github.com/wenhao-gao/mol_opt)<sup>4</sup>, is contained in the [gp folder](gp/). 

<sup>1</sup> Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data, 1(1):140022, August 2014. doi: 10.1038/sdata.2014.22.

<sup>2</sup> Lars Ruddigkeit, Ruud van Deursen, Lorenz C. Blum, and Jean-Louis Reymond. Enumeration of 166 Billion Organic Small Molecules in the Chemical Universe Database GDB-17. Journal of Chemical Information and Modeling, 52(11):2864–2875, November 2012. doi: 10.1021/ci300415d.

<sup>3</sup> Felix Wong, Erica J. Zheng, Jacqueline A. Valeri, Nina M. Donghia, Melis N. Anahtar, Satotaka Omori, Alicia Li, Andres Cubillos-Ruiz, Aarti Krishnan, Wengong Jin, Abigail L. Manson, Jens Friedrichs, Ralf Helbig, Behnoush Hajian, Dawid K. Fiejtek, Florence F. Wagner, Holly H. Soutter, Ashlee M. Earl, Jonathan M. Stokes, Lars D. Renner, and James J. Collins. Discovery of a structural class of antibiotics with explainable deep learning. Nature, 626(7997):177–185, February 2024. doi: 10.1038/s41586-023-06887-8.

<sup>4</sup> Wenhao Gao, Tianfan Fu, Jimeng Sun, and Connor W. Coley. Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, June 2022.