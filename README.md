# Prototype-enhanced Spatial Attention Network for Spatio-temporal Data Imputation

This is a pytorch implementation of 《Prototype-enhanced Spatial Attention Network for Spatio-temporal Data Imputation》


## Abstract

...

## Performance

...

## Architecture

```
<TBD>
```

## Installation

We provide a conda environment with all the project dependencies. To install the environment use:

```
conda env create -f conda_env.yml
conda activate psan
```

## Project Structure

..

## Experiments

The experiment can be performed by following the commands below.

* The script `run_spatiotemporal_imputation.py` is utilized for training PSAN. Below is an example:
```angular2html
 conda activate psan
 python ./experiments/run_spatiotemporal_imputation.py --config imputation/psan_la.yaml --model-name psan --dataset-name la_point
```
* The script `run_inference.py` is used to replicate results with a pre-trained model. Below is an example:
```angular2html
 conda activate psan
 python ./experiments/run_inference.py --config inference.yaml --model-name psan --dataset-name la_point --exp-name {exp_name}
```

## Citation

```
<TBD>
```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm