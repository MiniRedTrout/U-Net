import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.enc1 = self._contract(in_channel, 64)
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
        
        self._init_weights()
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
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
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
