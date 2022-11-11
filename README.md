# gnn-ops-benchmark
Profiling GNNs for scientific computing and benchmarking Pytorch and Pytorch Geometric operations relevant to SOTA graph neural network architectures. Accompanying code for [Operation-Level Performance Benchmarking of Graph Neural Networks for Scientific Applications](https://arxiv.org/pdf/2207.09955.pdf)

## Repo organization

This repository is organized into three high-level sections: 

1. A profiling module that supports end-to-end profiles of selected Pytorch Geometric GNN architectures
2. Several benchmark scripts for selected Pytorch and Pytorch Geometric low-level ops
3. Raw data and associated visualizaitons from benchmarking the above ops on NVIDIA A100 GPU

The following sections highlight usage of each of these features. 

## Getting Started

### Prerequisites and Installation

The following instructions should get the repository up and running on your local (UNIX) machine. 

First, create and activate a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Install pytorch and pytorch geometric:
```
pip install torch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu{CUDA_VERSION}.html
```
where `TORCH_VERSION` is your current version of pytorch and `CUDA_VERSION` is your current version of cuda. If you do not know what version of these libraries you have installed locally, you can find them by:

```
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

Next, you should install the remaining required libraries:

```
pip install -r requirements.txt
```

Confirm everything is working correctly by importing one of the profile modules:

```
python -c "from graph_benchmark.profile.OpProfiler import OpProfiler"
```

### Hardware Support

Currently, both the profiling module and ops-level benchmarking script only support CUDA enabled NVIDIA devices. 


## Usage: Profiler module 

The primary interface to the profiler module is the `graph_benchmark.profile.OpProfiler.OpProfiler`. This is the highest level class in the module, that incorporates features from `graph_benchmark.module` and `graph_benchmark.datasets` submodules. These submodules are also publically exposed endpoints and can thus be invoked directly. However, their use is outside the scope of this document. 

To use the `OpProfiler` class, relevant information need be stored in a `.json` file. This file will contain relevant information about each model to be profiled, its associated hyperparameters, as well as settings for the profiler. An example of such a json file is `prof_config.json` in the root of the project repository. 

After setting up the config file, one can simply invoke an instance of the `OpProfiler` class in order to profile any supported model. The full list of supported models can be found in `graph_benchmark.datasets.ptg_models`. After profiling, relevant data will be written to `.log`. These can then be visualized with Tensorboard. For instructions on using Tensorboard, please refer to this [tutorial](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html). 

## Usage: Benchmark Scripts

Currently, benchmark scripts are individual to each operation and can be invoked directly. All scripts are contained in `./op_bm_scripts`. To run a single script, for example for the `torch_scatter.scatter_min` op, simply invoke the corresponding script:

```
python benchmark_scatter_min.py
```
The relevant data will then be written to `./data` directory.


## Raw data
Raw data from benchmarking all operations in the `./op_bm_scripts` directory on NVIDIA A100 GPU (at >95% total DDR capacity) can be found in this repository's `./data` directory. Data is in `.csv` format. 

## Operation benchmarking visualizations

Visualizations for each of these operations are found in the `./notebooks/gnn_bm_analysis.ipynb` notebook. The notebook need not be rerun if one is interested in the data from our benchmarking. Simply view the notebook outputs, either locally, or online through the github interface. 

## Acknowledgements

This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

# Citing this work
If you found this work useful for your research, please cite our workshop paper using the following Bibtex snippet:
```
@misc{https://doi.org/10.48550/arxiv.2207.09955,
  doi = {10.48550/ARXIV.2207.09955},
  
  url = {https://arxiv.org/abs/2207.09955},
  
  author = {Hosseini, Ryien and Simini, Filippo and Vishwanath, Venkatram},
  
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Hardware Architecture (cs.AR), Performance (cs.PF), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Operation-Level Performance Benchmarking of Graph Neural Networks for Scientific Applications},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```


