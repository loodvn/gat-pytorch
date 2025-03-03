from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.pattern_gat import PatternGAT
from models.planetoid_gat import PlanetoidGAT
from models.ppi_gat import PPI_GAT


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


def load(config, file_name_ending='-best.ckpt', checkpoint_path=None):
    path = 'checkpoints/'+config['dataset'] + file_name_ending
    if checkpoint_path is not None:
        path = checkpoint_path

    if config['dataset'] == 'PPI':
        loaded_model = PPI_GAT.load_from_checkpoint(checkpoint_path=path, **config, strict=False)
    elif config['dataset'] == 'PATTERN':
        loaded_model = PatternGAT.load_from_checkpoint(checkpoint_path=path, **config)
    else:
        loaded_model = PlanetoidGAT.load_from_checkpoint(checkpoint_path=path, **config)
    return loaded_model