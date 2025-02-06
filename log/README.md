# Prototype-enhanced Spatial Attention Network for Robust Spatio-temporal Data Imputation

This is a pytorch implementation of "Prototype-enhanced Spatial Attention Network for Spatio-temporal Data Imputation" for EDBT2026


## Abstract

Spatio-temporal data analysis plays a critical role in air quality monitoring, traffic prediction and weather forecasting, yet real-world data often suffer from incompleteness. Existing imputation methods address missing values by spatio-temporal dependency modeling, but traditional autoregressive approaches are prone to error accumulation due to missing temporal frames. While some models mitigate errors through predefined graph structures and distributional assumptions, they fail to capture latent dependencies among nodes or complex data distribution patterns. To address these challenges, we propose a Prototype-enhanced Spatial Attention Network (PSAN). The proposed method effectively extracts both global and local temporal features and employs a bidirectional convolutional structure to mitigate the effects of missing time frames. Moreover, the proposed prototype learning module dynamically learns latent dependencies among nodes and adjusts the spatial attention as a matrix to reweight node relationships instead of a static graph structure. The experimental evaluations exhibit that the proposed PSAN outperforms state-of-the-art methods on four public spatio-temporal traffic datasets and demonstrates robust imputation performance even in scenarios with severe missing data.

## Performance

```
Please refer the submitted main file.
```

## Architecture

- **config/imputation/**: This directory contains the configuration information for the model.
- **experimes/models/models**: This directory contains the main model code.
- **experimes/run_spatiotemporal_imputation.py**: training file
- **experimes/run_inference.py**: testing file
- **log/**: this directory contains pre-trained model files
- **conda.yaml**: dependency environment.

## Installation

We provide a conda environment with all the project dependencies. To install the environment use:

```shell
conda env create -f conda_env.yml
conda activate psan
```

## Project Structure

..

## Experiments

The experiment can be performed by following the commands below.

* The script `run_spatiotemporal_imputation.py` is utilized for training PSAN. Below is an example:
```shell
 conda activate psan
 python ./experiments/run_spatiotemporal_imputation.py --config imputation/psan_la.yaml --model-name psan --dataset-name la_point
```
* The script `run_inference.py` is used to replicate results with a pre-trained model. Below is an example:
```shell
 conda activate psan
 python ./experiments/run_inference.py --config inference.yaml --model-name psan --dataset-name la_point --exp-name {exp_name}
```

## Citation

```
<TBD>
```

## Trained Model

```http
We provide the trained model via GitHub:
https://github.com/CQRhinoZ/PSAN
```



Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm