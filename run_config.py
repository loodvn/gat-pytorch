
# remember to set values of in and output to features and classes in GAT
data_config = {
    "PPI": {
        "test_type": "Inductive",
        "layer_type": "PyTorch_Geometric",
        "num_input_node_features": 50,
        "num_layers": 3, 
        "num_heads_per_layer": [4, 4, 6],  
        "heads_concat_per_layer": [True, True, False],
        "head_output_features_per_layer": [50, 256, 256, 121],
        "num_classes": 121,
        "add_skip_connection": True, 
        "dropout": 0.0,
        "l2_reg": 0.0, 
        "learning_rate": 0.005,
        "train_batch_size": 2,
        "num_epochs": 1000
        # Do we need to add bias.
    },
    "Cora": {
        "test_type": "Transductive",
        "layer_type": "PyTorch_Geometric",
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
        "num_epochs": 1000
        # Do we need to add bias.
    }
}