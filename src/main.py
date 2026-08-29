# training script for lightning model
"""
run via:
(changing data and model)
python main.py data=data/data3.yaml model=model/model3.yaml

to test:
python main.py skip_training=true
"""
import pytorch_lightning as pl
import torch
from torch import nn
import pandas as pd
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import json
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path

@hydra.main(config_path="conf", config_name="config")
def train_model(cfg):
    # set seed
    pl.seed_everything(cfg.seed)

    print(cfg)
    if 'memory' in cfg['model_module'].keys(): # if is key
        memory = cfg['model_module']['memory']
        cfg.data_module.memory = memory

    data_module = instantiate(cfg.data_module)

    # correct dim
    x_dim = data_module.dims[0]
    if 'dim' in cfg.model_module.keys():
        cfg.model_module.dim = x_dim
    elif 'input_dim' in cfg.model_module.keys():
        cfg.model_module.input_dim = x_dim
        cfg.model_module.output_dim = x_dim


    if 'treatment_cond' in cfg.model_module.keys():
        # for conditional models, need this to configure
        cfg.model_module.treatment_cond = len(data_module.cond_headings)

    model = instantiate(cfg.model_module)

    # conditional models need train_consecutive false!
    if not('Cond' in model.naming):
        cfg.data_module.train_consecutive = True
        data_module = instantiate(cfg.data_module)
    else:
        cfg.data_module.train_consecutive = False
        data_module = instantiate(cfg.data_module)
    
    wandb_config = {key: value for key, value in cfg.model_module.items() if key not in ['_target_']}
    wandb_config['model'] = model.naming
    wandb_config['data'] = data_module.naming
    wandb_config['mode'] = 'batch_run'
    wandb_config['x_headings'] = data_module.x_headings
    wandb_config['cond_headings'] = data_module.cond_headings
    wandb_config['t_headings'] = data_module.t_headings
    wandb_config['seed'] = cfg.seed

    wandb_logger = None
    wandb_run = None
    if cfg.wandb_logging and not(cfg.skip_training):
        wandb_savedir = to_absolute_path(cfg.wandb_dir)
        os.makedirs(wandb_savedir, exist_ok=True)
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            dir=wandb_savedir,
            config=wandb_config,
        )
        wandb_logger = WandbLogger(experiment=wandb_run)

    ckpt_savedir = os.path.join(
        to_absolute_path(cfg.ckpt_dir),
        model.naming + '_' + data_module.naming,
    )
    os.makedirs(ckpt_savedir, exist_ok=True)

    results_savedir = os.path.join(
        to_absolute_path(cfg.results_dir),
        model.naming + '_' + data_module.naming,
    )
    os.makedirs(results_savedir, exist_ok=True)

    resume_ckpt_path = (
        to_absolute_path(cfg.resume_ckpt_path)
        if cfg.resume_ckpt_path is not None
        else None
    )

    print(f"[experiment] seed={cfg.seed}")
    print(f"[experiment] checkpoint_dir={ckpt_savedir}")
    print(f"[experiment] results_dir={results_savedir}")
    print(f"[experiment] resume_ckpt_path={resume_ckpt_path}")

    resolved_config_path = os.path.join(results_savedir, 'resolved_config.yaml')
    with open(resolved_config_path, 'w', encoding='utf-8') as config_file:
        config_file.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"[experiment] resolved_config={resolved_config_path}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_savedir,
        filename='best_model',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    recovery_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_savedir, 'recovery'),
        filename='recovery-{epoch:04d}-{step}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=5,
        save_on_train_epoch_end=True,
        save_weights_only=False,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        max_time=cfg.max_time, 
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        callbacks=[
            checkpoint_callback,
            recovery_checkpoint_callback,
            early_stopping_callback,
        ],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger if wandb_logger is not None else False,
        limit_train_batches=0 if cfg.skip_training else cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        strategy='auto',
    )

    # Train the model
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_ckpt_path,
    )

    # Preserve the original evaluation behavior: test the final in-memory model.
    test_results = trainer.test(model, datamodule=data_module)

    result_payload = {
        'seed': int(cfg.seed),
        'model_naming': model.naming,
        'data_naming': data_module.naming,
        'resume_ckpt_path': resume_ckpt_path,
        'resumed': resume_ckpt_path is not None,
        'test_results': test_results[0] if len(test_results) == 1 else test_results,
    }
    result_path = os.path.join(
        results_savedir,
        f'final_test_results_seed_{cfg.seed}.json',
    )
    with open(result_path, 'w', encoding='utf-8') as result_file:
        json.dump(result_payload, result_file, indent=2, sort_keys=True)

    print(f"[experiment] final_test_results={result_path}")

    if wandb_run is not None:
        wandb.finish()

def main():
    train_model()

if __name__ == '__main__':
    main()
