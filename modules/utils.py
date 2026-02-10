import torch
from torch.utils.data import DataLoader
from clearml import Task
from omegaconf import OmegaConf
from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex, Precision, Recall

def settings(cfg):
    import sys
    sys.path.append('/content/drive/MyDrive/segmentation_project')
    from modules.config import load_config
    from modules.data import split, Dataset, transformers
    from modules.model import UNet
    
    task = Task.init(
        project_name='JustS',
        task_name=f'UNet_20epochs',
        auto_connect_frameworks={'pytorch': True, 'hydra': True}
    )
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect_configuration(cfg_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = split()
    train_transform, val_transform = transformers()
    
    train_dataset = Dataset(
        images_dir=paths['train_images'],
        labels_dir=paths['train_labels'],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    
    test_dataset = Dataset(
        images_dir=paths['test_images'],
        labels_dir=paths['test_labels'],
        transform=val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    model = UNet(cfg.model.in_channel, cfg.model.out_channel).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    
    metrics = {
        'dice': DiceScore(num_classes=2, average='macro', input_format='index').to(device),
        'iou': JaccardIndex(num_classes=2, task='multiclass', average='macro').to(device),
        'precision': Precision(num_classes=2, task='multiclass', average='macro').to(device),
        'recall': Recall(num_classes=2, task='multiclass', average='macro').to(device)
    }
    
    return {
        'device': device,
        'model': model,
        'test_loader': test_loader,
        'train_loader': train_loader,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'metrics': metrics,
        'task': task
    }
