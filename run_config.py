from enum import Enum


class LayerType(Enum):
    GATLayer = 1
    PyTorch_Geometric = 2


class Dataset(Enum):
    PPI = 1
    Cora = 2
    Citeseer = 3
    Pubmed = 4


# remember to set values of in and output to features and classes in GAT
data_config = {
    "PPI": {
        "test_type": "Inductive",
        "layer_type": LayerType.GATLayer,
        # "layer_type": LayerType.PyTorch_Geometric,
        "num_input_node_features": 50,
        "num_layers": 3, 
        "num_heads_per_layer": [4, 4, 6],  
        "heads_concat_per_layer": [True, True, False],
        "head_output_features_per_layer": [50, 256, 256, 121],
        "num_classes": 121,
        "add_skip_connection": [False, True, False],
        "dropout": 0.0,
        "l2_reg": 0.0, 
        "learning_rate": 0.005,
        "train_batch_size": 2,
        "num_epochs": 1000,
        "const_attention": False
        # Do we need to add bias.
    },
    "Cora": {
        "test_type": "Transductive",
        "layer_type": LayerType.GATLayer,
        # "layer_type": LayerType.PyTorch_Geometric,
        "num_layers": 2, 
        "num_input_node_features": 1433,
        "num_heads_per_layer": [8, 1],  
        "heads_concat_per_layer": [True, False],
        "head_output_features_per_layer": [1433, 8, 7],  
        "num_classes": 7,
        "add_skip_connection": False, 
        "dropout": 0.6,
        "l2_reg": 0.0005, 
        "learning_rate": 0.005,
        "train_batch_size": 1,
        "num_epochs": 1000,
        "const_attention": False
        # Do we need to add bias.
    },
    "Citeseer": {
        "test_type": "Transductive",
        "layer_type": LayerType.GATLayer,
        # "layer_type": LayerType.PyTorch_Geometric,
        "num_layers": 2, 
        "num_input_node_features": 3703,
        "num_heads_per_layer": [8, 1],  
        "heads_concat_per_layer": [True, False],
        "head_output_features_per_layer": [3703, 8, 6],  
        "num_classes": 6,
        "add_skip_connection": False, 
        "dropout": 0.6,
        "l2_reg": 0.0005, 
        "learning_rate": 0.005,
        "train_batch_size": 1,
        "num_epochs": 1000,
        "const_attention": False
        # Do we need to add bias.
    },
    "Pubmed": {
        "test_type": "Transductive",
        "layer_type": LayerType.GATLayer,
        # "layer_type": LayerType.PyTorch_Geometric,
        "num_layers": 2, 
        "num_input_node_features": 500,
        "num_heads_per_layer": [8, 8],  
        "heads_concat_per_layer": [True, False],
        "head_output_features_per_layer": [500, 8, 3],  
        "num_classes": 3,
        "add_skip_connection": False, 
        "dropout": 0.6,
        "l2_reg": 0.001, 
        "learning_rate": 0.01,
        "train_batch_size": 1,
        "num_epochs": 1000,
        "const_attention": False
        # Do we need to add bias.
    }
}