# BIND: Binding Intertemporal Nodes for Multivariate Timeseries Forecasting



## Introduction

The repository is for paper "BIND: Binding Intertemporal Nodes for Multivariate Timeseries Forecasting". 

## Abstract
The multivariate time series forecasting task can be expressed as a process of inferring the future state of multi-variable based on multivariate historical data through spatiotemporal data fusion. The spatial dimension of spatiotemporal data represents variables in a multivariate time series. Multivariate time series forecasting tasks have important practical significance for assisting decision-making, avoiding risks, and improving returns. Although there are many studies on forecasting tasks based on multivariate time series data, most of the existing studies adopt the perspective of separating the time dimension and the space dimension, that is, adopting a relatively independent time-dependent and space-dependent learning model. In this light, we propose intertemporal spatial correlation, a novel concept that characterizes the complicated functional ties between different points at different times, to incorporate spatial and temporal semantics simultaneously for multi-variate timeseries forecasting, which is incorporated in to BIND (Binding Intertemporal NoDes), a novel framework that features leveraging a Spatial Temporal Attention (S-T Atten) module and a Kronecker product-based Kronecker and Attention Graph Generator (K&AGG) module to generate both conditional and unconditional time-aware adjacency matrices for learning timely yet stable intertemporal spatial correlations. Experiments on 4 benchmark datasets demonstrate the significant improvements of BIND over the SOTAs. Extra studies illustrate the distinct functions of K&AGG and S-T Atten modules and their synergistic effect, and visualize that BIND indeed can find the intertemporal correlations between spatial points with phase differences and is more robust to coarse time granularity compared with competitors.

## Framework
![image](https://github.com/WU-Guanying/BIND-main/blob/main/Figs/STRUCTURE.eps)


## Citation
If you find this code useful, you may cite us as:

```

```

## Setup Python environment for BIND
Install python environment
```{bash}
$ conda env create -f environment.yml 
```


## Reproducibility
### Usage
#### In terminal
- Run the shell file (at the root of the project)

```{bash}
$ bash run.sh
```
- Run the python file (at the `model` folder)
```{bash}
$ cd model

$ python Run.py --dataset='PEMSD8' 
```
