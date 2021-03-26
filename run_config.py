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
        "layer_type": LayerType.GATLayer,
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
        "batch_size": 2,
        "num_epochs": 1000,
        "const_attention": False
    },
    "PATTERN": {
        "layer_type": LayerType.GATLayer,
        "num_input_node_features": 3,
        "num_layers": 4, 
        "num_heads_per_layer": [4, 4, 4, 1],  
        "heads_concat_per_layer": [True, True, True, False],
        "head_output_features_per_layer": [3, 12, 24, 12, 1],
        "num_classes": 1,
        "add_skip_connection": [True, True, True, True], 
        "dropout": 0,
        "l2_reg": 0, 
        "learning_rate": 0.005,
        "batch_size": 8,
        "num_epochs": 1000,
        "const_attention": False
    },
    "Cora": {
        "layer_type": LayerType.GATLayer,
        "num_layers": 2, 
        "num_input_node_features": 1433,
        "num_heads_per_layer": [8, 1],  
        "heads_concat_per_layer": [True, False],
        "head_output_features_per_layer": [1433, 8, 7],  
        "num_classes": 7,
        "add_skip_connection": [False, False],
        "dropout": 0.6,
        "l2_reg": 0.0005, 
        "learning_rate": 0.005,
        "batch_size": 1,
        "num_epochs": 1000,
        "const_attention": False
    },
    "Citeseer": {
        "layer_type": LayerType.GATLayer,
        "num_layers": 2, 
        "num_input_node_features": 3703,
        "num_heads_per_layer": [8, 1],  
        "heads_concat_per_layer": [True, False],
        "head_output_features_per_layer": [3703, 8, 6],  
        "num_classes": 6,
        "add_skip_connection": [False, False],
        "dropout": 0.6,
        "l2_reg": 0.0005, 
        "learning_rate": 0.005,
        "batch_size": 1,
        "num_epochs": 1000,
        "const_attention": False
    },
    "Pubmed": {
        "layer_type": LayerType.GATLayer,
        "num_layers": 2, 
        "num_input_node_features": 500,
        "num_heads_per_layer": [8, 8],  
        "heads_concat_per_layer": [True, False],
        "head_output_features_per_layer": [500, 8, 3],  
        "num_classes": 3,
        "add_skip_connection": [False, False],
        "dropout": 0.6,
        "l2_reg": 0.001, 
        "learning_rate": 0.01,
        "batch_size": 1,
        "num_epochs": 1000,
        "const_attention": False
    }
}