from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.trans_gat import transGAT
from models.indu_gat import induGAT


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def checkpoint(filename, monitor='val_loss', dirpath='checkpoints', mode='min',):
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=dirpath,
        filename=filename,
        mode=mode,
    )
    return checkpoint_callback


def early_stop(monitor='val_loss', patience=100, verbose=True, mode='min'):
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        mode=mode
    )
    return early_stop_callback

def load(config, file_name_ending):
    if config['test_type'] == 'Inductive':
        loaded_model = induGAT.load_from_checkpoint(checkpoint_path='checkpoints/'+config['dataset'] + file_name_ending, **config)
    else:
        loaded_model = transGAT.load_from_checkpoint(checkpoint_path ='checkpoints/'+config['dataset'] + file_name_ending, **config)
    return loaded_model