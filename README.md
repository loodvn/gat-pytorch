# Graph Attention Networks (GAT) - PyTorch
Reproducing Graph Attention Networks by Velickovic et al.

By Oliver Warwick, Pavlos Piperis, Khalil Ibrahim and Lood van Niekerk, for Advanced ML class 2021.

This repo contains our implementation of the GAT layer along with code to run all our training, testing, visualisation and other experiments.

The information below details how to get the code running locally via the command line. However, we strongly recommend using the jupyter notebook (Reproduce_Experiments.ipynb) to do this instead (through Colab to use GPUs). It is both simpler and more visually appealing, whilst still being able to display the complete functionality of our solution.

## Setup

Run the following commands to set up a conda environment. Given that the whole team use Mac, we only have the necessary setup file for this.

```
conda env create -f env/gat_req_mac_version.yml
conda activate ATML_HT
```

For any other platform, you will have to install the necessary packages manually using the requirements below.

### Requirements
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

The commands for running the files follow the general structure of:

`python <filename>.py --dataset <dataset> --<other_flag_keys> <other_flag_val>`

Where the filname is either train or vis, and the dataset is any which is in our config file. 

When using vis, the only other valid flag is 'vis_type' which must be Neighbourhood, Weight, or Entropy.

When using train, the runtime parameters of 'num_epochs', 'l2_reg', 'learning_rate' and 'patience' can all be used, though are not necessary as there are default values in the config file. Additionally, 'exec_type' can be set to 'load' in order to skip training and test a pretrained model.

### Visualisation Examples (Generated on the PATTERN dataset)
Neighbourhood Plot:

![PPI_Neigh_2](https://user-images.githubusercontent.com/25391634/114173657-66138500-992f-11eb-8d34-7d8d26cd565b.png)

Entropy Histogram:

![pattern_entropy](https://user-images.githubusercontent.com/25391634/114173249-c5bd6080-992e-11eb-8122-e16dc4c45cb1.png)

Weight Histogram:

![pattern_attention_weights](https://user-images.githubusercontent.com/25391634/114173239-c229d980-992e-11eb-8fc2-7da19e4b8dfa.png)


## Credits
- The original TF repo by Petar Velickovic, author of Graph Attention Networks: https://github.com/PetarV-/GAT (used to check implementation details not found in paper)
- PyTorch implementation of GAT, endorsed by Petar Velickovic: https://github.com/gordicaleksa/pytorch-GAT (had many useful functions that we used for models/gat_layer.py)
- DGL (Deep Graph Library) blogpost on GAT: https://www.dgl.ai/blog/2019/02/17/gat.html (used for attention weight visualisations)

Specific acknowledgements are made inside the codebase where outside code is used.
