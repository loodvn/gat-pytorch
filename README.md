# Graph Attention Networks (GAT) - PyTorch
Reproducing Graph Attention Networks by Velickovic et al.
By Oliver Warwick, Pavlos Piperis, Khalil Ibrahim and Lood van Niekerk

This repo contains all of the code for our implentation of the GAT layer and everything needed to run it.

The information below details how to get the code running locally via the command line. However, we strongly recommend using the jupyter notebook (Reproduce_Experiments.ipynb) to do this instead (through Colab to use GPUs). 
It is both simpler and more visually appealing, whilst still being able to display the complete functionality of our solution.

## Setup
Run the following commands to set up a conda environment.
Given that the whole team use Mac, we only have the necessary setup file for this. 

```
conda env create -f env/gat_req_mac_version.yml
conda activate ATML_HT
```

## Requirements

```
PyTorch==1.7.0
PyTorch Lightning==1.2.2
PyTorch Geometric==1.6.3
pandas==1.2.1
numpy==1.19.2
PyCairo==1.20.0
```
PyTorch Geometric can be installed from https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.
Pycairo can be installed from https://pycairo.readthedocs.io/en/latest/getting_started.html

## Example Usage


TODO(Lood): train.py, vis.py etc

TODO(Lood) include plots

## Credits
- The original TF repo by Petar Velickovic, author of Graph Attention Networks: https://github.com/PetarV-/GAT (used to check implementation details not found in paper)
- PyTorch implementation of GAT, endorsed by Petar Velickovic: https://github.com/gordicaleksa/pytorch-GAT (had many useful functions that we used for models/gat_layer.py)
- DGL (Deep Graph Library) blogpost on GAT: https://www.dgl.ai/blog/2019/02/17/gat.html (used for attention weight visualisations)

Specific acknowledgements are made inside the codebase where outside code is used.
