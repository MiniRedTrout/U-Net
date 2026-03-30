import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from clearml import Logger, Task
from torchmetrics import JaccardIndex, Precision, Recall
from torchmetrics.segmentation import DiceScore

from modules.loss import Loss


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.enc1 = self._contract(in_channel,64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._contract(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._contract(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 2, stride=2)
        )
        
        self.dec3 = self._exp(512, 256, 128)
        self.dec2 = self._exp(256, 128, 64)
        self.fc = self._final(128, 64, out_channel)
    
    def _contract(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def _exp(self, in_channels, hidden_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.ConvTranspose2d(hidden_channels, out_channels, 2, stride=2)
        )
    
    def _final(self, in_channels, hidden_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        )
    
    def _c_c(self, up, down):
        _, _, hu, wu = up.shape
        _, _, hb, wb = down.shape
        ch = (hb - hu) // 2
        cw = (wb - wu) // 2
        cropped = down[:, :, ch:ch+hu, cw:cw+wu]
        return torch.cat([up, cropped], dim=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        d3 = self._c_c(b, e3)
        c2 = self.dec3(d3)
        d2 = self._c_c(c2, e2)
        c1 = self.dec2(d2)
        d1 = self._c_c(c1, e1)
        out = self.fc(d1)
        return out

class LightUNet(L.LightningModule):
    def __init__(self,config,task):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 
        self.model = UNet(config.model.in_channel,config.model.out_channel)
        self.criterion = Loss()
        self.best_val_epoch = 0
        self.task = task 
        self.val_dice = DiceScore(num_classes=2, average='macro')
        self.train_dice = DiceScore(num_classes=2, average='macro')
        self.iou = JaccardIndex(num_classes=2, average='macro',task='multiclass')
        self.precision = Precision(num_classes=2, average='macro',task='multiclass')
        self.recall = Recall(num_classes=2, average='macro',task='multiclass')
    def forward(self, x):
        return self.model(x)
    def training_step(self,batch,batch_idx):
        images,labels = batch 
        labels = labels.long()
        out = self(images)
        loss = self.criterion(out,labels)
        preds = torch.argmax(out,dim=1)
        dice = self.train_dice(preds,labels)
        if batch_idx % 10 == 0:
            logger = self.task.get_logger()
            logger.report_scalar('train','loss',loss.item(),iteration=self.global_step)
            logger.report_scalar('train','dice',dice.item(),iteration=self.global_step)
        return loss 
    def validation_step(self,batch,batch_idx):
        images,labels = batch 
        labels = labels.long()
        out = self(images)
        loss = self.criterion(out,labels)
        preds = torch.argmax(out,dim=1)
        dice = self.val_dice(preds,labels)
        iou = self.iou(preds,labels)
        precision = self.precision(preds,labels)
        recall = self.recall(preds,labels)
        logger = self.task.get_logger()
        logger.report_scalar(
            'val',
            'loss',
            loss.item(),
            iteration=self.current_epoch
        )
        logger.report_scalar(
            'val',
            'dice',
            dice.item(),
            iteration=self.current_epoch
        )
        logger.report_scalar(
            'val',
            'iou',
            iou.item(),
            iteration=self.current_epoch
        )
        logger.report_scalar(
            'val',
            'precision',
            precision.item(),
            iteration=self.current_epoch
        )
        logger.report_scalar(
            'val',
            'recall',
            recall.item(),
            iteration=self.current_epoch
        )
        self.log(
            'val_loss',
            loss,
            on_step=False, 
            on_epoch=True, 
            prog_bar=True
        )
        self.log(
            'val_dice', 
            dice, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True
        )
        self.log(
            'val_iou',
            iou, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True
        )
        self.log(
            'val_precision', 
            precision, 
            on_step=False,
            on_epoch=True, 
            prog_bar=True
        )
        self.log(
            'val_recall',
            recall, 
            on_step=False,
            on_epoch=True, 
            prog_bar=True
        )
        return {
            'val_loss':loss,
            'val_dice':dice,
            'iou':iou,
            'precision':precision,
            'recall':recall
        }
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True}}
    def on_train_epoch_end(self):
        self.train_dice.reset()
    def on_validation_epoch_end(self):
        self.val_dice.reset()
        self.iou.reset()
        self.precision.reset()
        self.recall.reset()
