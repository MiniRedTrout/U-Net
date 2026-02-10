import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex, Precision, Recall

def train_epoch(model,train_loader,criterion,optimizer,device):
  model.train()
  epoch_train_loss = 0
  train_pbar = tqdm(train_loader,desc='Train')
  for idx, (images, labels) in enumerate(train_pbar):
    images, labels = images.to(device), labels.to(device)
    labels = labels.long()
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    epoch_train_loss += loss.item()
    train_pbar.set_postfix({
        'Loss': f'{loss.item():.4f}',
        'Avg Loss': f'{epoch_train_loss/(idx+1):.4f}'
    })
  train_pbar.close()
  avg_train_loss = epoch_train_loss / len(train_loader)
  return avg_train_loss

def validate_epoch(model, test_loader, metrics, device):
    model.eval()
    for metric in metrics.values():
        metric.reset()
    val_pbar = tqdm(test_loader, desc='[Val]')
    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_pbar):
            images, masks = images.to(device), masks.to(device)
            masks = masks.long()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            for metric in metrics.values():
                metric.update(preds, masks)
    val_pbar.close()
    val_metrics = {}
    for name, metric in metrics.items():
        val_metrics[name] = metric.compute().item()

    return val_metrics

def early_stopping(cur_metric, best_metric, patience_count, patience, min_d, epoch, model):
    better = False
    if cur_metric < best_metric - min_d:
      better = True
    else:
      if cur_metric > best_metric + min_d:
            better = True
    if better:
        best_metric = cur_metric
        patience_count = 0
        best_epoch = epoch
        best_model_weights = model.state_dict().copy()
    else:
        patience_count += 1
        best_model_weights = None
    should_stop = patience_count >= patience
    return should_stop, best_metric, patience_count, best_model_weights

def train_model():
    import sys
    sys.path.append('/content/drive/MyDrive/segmentation_project')
    from modules.config import config
    from modules.utils import settings
    
    cfg = config()
    setup = settings(cfg)
    
    device = setup['device']
    model = setup['model']
    train_loader = setup['train_loader']
    test_loader = setup['test_loader']
    criterion = setup['criterion']
    optimizer = setup['optimizer']
    scheduler = setup['scheduler']
    metrics = setup['metrics']
    task = setup['task']
    es_enabled = cfg.early_stopping.enabled
    patience = cfg.early_stopping.patience
    min_delta = cfg.early_stopping.min_delta
    restore_best = cfg.early_stopping.restore_best_weights
    best_dice = 0
    best_epoch = 0
    patience_count = 0
    best_model_weights = None

    history = {
        'train_losses': [],
        'val_dice_scores': [],
        'val_iou_scores': [],
        'val_pres_scores': [],
        'val_rec_scores': []
    }
    logger = task.get_logger()
    for epoch in range(cfg.training.epochs):
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, test_loader, metrics, device)
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
              'dice_score': metrics['dice'],
              'iou_score': metrics['iou'],
              'precision': metrics['precision'],
              'recall': metrics['recall'],
              'loss': avg_train_loss
            },'best_model.pth')
            
        logger.report_scalar("Loss", "Train", train_loss, epoch)
        logger.report_scalar("Metrics", "Dice", val_metrics['dice'], epoch)
        logger.report_scalar("Metrics", "Iou", val_metrics['iou'], epoch)
        logger.report_scalar("Metrics", "Precision", val_metrics['precision'], epoch)
        logger.report_scalar("Metrics", "Recall", val_metrics['recall'], epoch)
        history['train_losses'].append(avg_train_loss)
        history['val_dice_scores'].append(val_metrics['dice'])
        history['val_iou_scores'].append(val_metrics['iou'])
        history['val_pres_scores'].append(val_metrics['precision'])
        history['val_rec_scores'].append(val_metrics['recall'])

        scheduler.step(avg_train_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}:')
        print(f'  Loss: {avg_train_loss:.4f}')
        print(f'  Dice: {val_metrics["dice"]:.4f}')
        print(f'  Iou: {val_metrics["iou"]:.4f}')
        print(f'  Precision: {val_metrics["precision"]:.4f}')
        print(f'  Recall: {val_metrics["recall"]:.4f}')
        print(f'  LR: {lr:.2e}')
        print(f'Best Dice {best_dice:.4f} Epoch {best_epoch}')
        should_stop, best_dice, patience_counter, best_model_weights = early_stopping(
                val_metrics['dice'],
                best_metric=best_dice,
                patience_count=patience_count,
                patience=patience,
                min_d=min_delta,
                epoch=epoch,
                model=model
        )
        if should_stop:
            print(f"Early stopping")
            if restore_best and best_model_weights is not None:
                model.load_state_dict(best_model_weights)
            break
    print(f"Best epoch: {best_epoch} Dice: {best_dice:.4f}")
    return model, history

