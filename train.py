import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.gat_layer import GATLayer
from models.trans_gat import transGAT
from models.indu_gat import induGAT


def train(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads, save):
    if dataset == 'PPI':
        gat = induGAT(dataset=dataset, node_features=node_features, num_classes=num_classes, lr=learning_rate, l2_reg=l2_reg)
    else:
        gat = transGAT(dataset=dataset, node_features=node_features, num_classes=num_classes, out_heads=out_heads, lr=learning_rate, l2_reg=l2_reg)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=dataset + '-best',#{epoch:02d}-{val_loss:.2f}')
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback, early_stop_callback], show_progress_bar=True)
    
    trainer.fit(gat)
    trainer.test(gat)
    checkpoint_callback.best_model_path
    checkpoint_callback.best_model_score
    trainer.test()

    if dataset == 'PPI':
        loaded_model = induGAT.load_from_checkpoint(checkpoint_path = 'checkpoints/'+ dataset + "-best.ckpt", dataset=dataset, node_features=node_features, num_classes=num_classes)
    else:
        loaded_model = transGAT.load_from_checkpoint(checkpoint_path = 'checkpoints/'+ dataset + "-best.ckpt", dataset=dataset, node_features=node_features, num_classes=num_classes)
    

    trainer.test(loaded_model)

    # if save:
    #     trainer.save_checkpoint(dataset + ".ckpt")
    #     print("Model has been saved")

# add if no checkpoint file then train
def load(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads):
    print('I am loading :)')
    loaded_model = transGAT.load_from_checkpoint(checkpoint_path ='checkpoints/'+dataset + "-best.ckpt", dataset=dataset, node_features=node_features, num_classes=num_classes)
    # Will need to change this as loaded dataset could be an induGAT
    # loaded_model = transGAT(dataset=dataset, node_features=node_features, num_classes=num_classes)
    # trainer = pl.Trainer(resume_from_checkpoint=dataset+".ckpt")
    trainer = pl.Trainer()
    # trainer.fit(loaded_model)
    trainer.test(loaded_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--max_epochs', default=100)
    parser.add_argument('--l2', default=0.0005)
    parser.add_argument('--lr', default=0.005)
    parser.add_argument('--exec_type', default='train') 

    parser.add_argument('--histograms', default=False)
    parser.add_argument('--save', default=False)

    args = parser.parse_args()
    dataset = args.dataset
    max_epochs = int(args.max_epochs)
    learning_rate = float(args.lr)
    l2_reg = float(args.l2)
    
    out_heads = 1
    
    if dataset == 'Cora':
        node_features = 1433
        num_classes = 7
    elif dataset == 'Citeseer':
        node_features = 3703
        num_classes = 6  
    elif dataset == 'Pubmed':
        node_features = 500
        num_classes = 3
        learning_rate = 0.01
        l2_reg = 0.001
        out_heads = 8
    elif dataset == 'PPI':
        node_features = 50
        num_classes = 121
        l2_reg = 0
    else:
        sys.exit('Dataset is invalid')

    if args.exec_type == 'train':
        train(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads, args.save)

    elif args.exec_type == 'load':
        load(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads)
    
