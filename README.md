# Graph Attention Networks (GAT) - PyTorch
Reproducing Graph Attention Networks by Velickovic et al.
By Oliver Warwick, Pavlos Piperis, Khalil Ibrahim and Lood van Niekerk

This repo contains all of the code for our implentation of the GAT layer and everything needed to run it.

The information below details how to get the code running locally via the command line. However, we strongly recommend using the jupyter notebook (Reproduce_Experiments.ipynb) to do this instead. It is both simpler and more visually appealing, whilst still being able to display the complete functionality of our solution.

## Setup
The necessary environmental setup can be done automatically through conda and the provided yaml file. Given that the whole team use Mac, we only have the necessary setup file for this. 

```
conda env create -f env/gat_req_mac_version.yml
conda activate ATML_HT
```

For any other environment, we recommend setting up the notebook and running shell commands from there.

## Example Usage

The commands for running the files follow the general structure of:

`python <filename>.py --dataset <dataset> --<other_flag_keys> <other_flag_val>`

Where the filname is either train or vis, and the dataset is any which is in our config file. 

When using vis, the only other valid flag is 'vis_type' which must be Neighbourhood, Weight, or Entropy.

When using train, the runtime parameters of 'num_epochs', 'l2_reg', 'learning_rate' and 'patience' can all be used, though are not necessary as there are default values in the config file. Additionally, 'exec_type' can be set to 'load' in order to skip training and test a pretrained model.

### Visualisation Examples
Neighbourhood Plot:

![PPI_Neigh_2](https://user-images.githubusercontent.com/25391634/114173657-66138500-992f-11eb-8d34-7d8d26cd565b.png)

Entropy Histogram:

![pattern_entropy](https://user-images.githubusercontent.com/25391634/114173249-c5bd6080-992e-11eb-8122-e16dc4c45cb1.png)

Weight Histogram:

![pattern_attention_weights](https://user-images.githubusercontent.com/25391634/114173239-c229d980-992e-11eb-8fc2-7da19e4b8dfa.png)

