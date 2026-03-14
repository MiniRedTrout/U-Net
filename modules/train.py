import pytorch_lightning as L 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint 
from clearml import Task
from omegaconf import OmegaConf
import os 
from modules.data import LightDataModule 
from modules.models.unet import LightUNet 
from modules.config import load_config
def train():
    cfg = load_config()
    task = Task.init(
        project_name='JustS',
        task_name=f'UNet_20epochs',
        auto_connect_frameworks={'pytorch': True, 'hydra': True}
    )
    task.connect(OmegaConf.to_container(cfg,resolve=True))
    data_module = LightDataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    model = LightUNet(cfg,task)
    callback = [
        EarlyStopping(
            monitor='val_dice',
            mode='max',
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta
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
        accelerator='auto',
        callbacks=callback,
        logger=False,
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )
    trainer.fit(model,data_module)
    task.update_output_model(
        model_path='checkpoints'
    )
    print(f"Лучший dice: {trainer.checkpoint_callback.best_model_score:.4f}")
    return model, trainer

if __name__ == '__main__':
    model, trainer = train()
