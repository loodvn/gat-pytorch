import argparse

import pytorch_lightning as pl

import data_utils as d_utils
from models.indu_gat import induGAT
from models.trans_gat import transGAT
from run_config import data_config


def run(config):
    checkpoint_callback = d_utils.checkpoint(filename=config['dataset']+'-best')

    early_stop_callback = d_utils.early_stop()

    # Use same random seed for reproducible, deterministic results
    # pl.seed_everything(42)

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=int(config['num_epochs']),
        callbacks=[checkpoint_callback, early_stop_callback],
        # deterministic=True,
        # weights_summary='full',
        # stochastic_weight_avg=True,
    )

    if config['exec_type'] == 'train':
        if config['test_type'] == 'Inductive':
            gat = induGAT(**config)
        else:
            gat = transGAT(**config)

        trainer.fit(gat)
        trainer.test(gat)
        checkpoint_callback.best_model_path
        trainer.test()
    else:
        try:
            gat = d_utils.load(config)
            trainer.test(gat)
            # weights = gat.attention_weights
        except FileNotFoundError:
            print("There is no saved checkpoint for this dataset!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--num_epochs') # should be max_epochs
    parser.add_argument('--l2_reg')
    parser.add_argument('--learning_rate')
    # this could throw an error if flag is set
    parser.add_argument('--patience')
    parser.add_argument('--exec_type', default='train')
    parser.add_argument('--histograms', default=False)

    args = parser.parse_args()
    dataset = args.dataset

    if dataset not in data_config.keys():
        print(f"Dataset not valid. Must be one of {data_config.keys()}. {dataset} given.")
    else:
        config = data_config[dataset]
        di = {k: v for k, v in args.__dict__.items() if v is not None}

        config.update(di)
        print(config)
        run(config)
