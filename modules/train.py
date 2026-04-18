
import pytorch_lightning as L
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from modules.data import LightDataModule
from modules.models.unet import LightUNet


def train(cfg: DictConfig):
    task = Task.init(
        project_name='JustS',
        task_name='UNet_20epochs',
        auto_connect_frameworks={'pytorch': True, 'hydra': True}
    )
    task.connect(OmegaConf.to_container(cfg,resolve=True))
    print('task')
    data_module = LightDataModule(cfg)
    print('data')
    data_module.prepare_data()
    print('prepare data')
    data_module.setup()
    print('setup')
    model = LightUNet(cfg,task)
    callback = [
        EarlyStopping(
            monitor='val_dice',
            mode='max',
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta
        ),
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='{epoch:02d}-{val_dice:.4f}',
            monitor='val_dice',
            mode='max',
            save_last=True
        )
    ]
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='gpu',
        devices=1,
        callbacks=callback,
        logger=True,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        log_every_n_steps=10,
        enable_model_summary=True
    )
    trainer.fit(model,data_module)
    task.update_output_model(
        model_path='checkpoints'
    )
    print(f"Лучший dice: {trainer.checkpoint_callback.best_model_score:.4f}")
    return model, trainer


