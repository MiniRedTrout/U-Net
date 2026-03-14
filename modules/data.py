import os 
import shutil 
import cv2 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from pathlib import Path

class Dataset(Dataset):
    def __init__(self,images_dir,labels_dir,transform):
        self.images_dir = images_dir 
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_f = sorted([f for f in os.listdir(images_dir) if f.endswith('_image.png')])
        self.label_f = sorted([f for f in os.listdir(labels_dir) if f.endswith('_label.png')])
    def __len__(self):
        return len(self.image_f)
    def __getitem__(self,idx):
        label_path = os.path.join(self.labels_dir,self.label_f[idx])
        image_path = os.path.join(self.images_dir,self.image_f[idx])
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        transformed = self.transform(image=image,mask=label)
        image = transformed['image']
        label = transformed['mask']
        label = (label > 0).long()
        return image,label 

class LightDataModule(L.LightningDataModule):
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.batch_size = config.training.batch_size 
        self.num_workers = config.data.num_workers
    def prepare_data(self):
        path = '/kaggle/input/2d-slicing-of-imagetbad-dataset'
        im_path = os.path.join(path, '2D dataset', 'images')
        lb_path = os.path.join(path, '2D dataset', 'labels')
        output = self.config.data.output
        pairs = []
        for file in os.listdir(im_path):
            if file.endswith('_image.png'):
                lb_file = file.replace('_image.png', '_label.png')
                if os.path.exists(os.path.join(lb_path, lb_file)):
                    pairs.append((file, lb_file))
        train_pairs, test_pairs = train_test_split(
            pairs,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state
        )
        for cat, pair in [('train', train_pairs), ('test', test_pairs)]:
            img_dir = os.path.join(output, cat, 'images')
            lbl_dir = os.path.join(output, cat, 'labels')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            for img_file, lbl_file in pair:
                shutil.copy2(os.path.join(im_path, img_file), os.path.join(img_dir, img_file))
                shutil.copy2(os.path.join(lb_path, lbl_file), os.path.join(lbl_dir, lbl_file))
        self.paths = {
            'train_images': os.path.join(output, 'train', 'images'),
            'train_labels': os.path.join(output, 'train', 'labels'),
            'test_images': os.path.join(output, 'test', 'images'),
            'test_labels': os.path.join(output, 'test', 'labels')
        }
    def setup(self,stage=None):
        train_transform = A.Compose([
            A.Resize(self.config.transforms.image_size, self.config.transforms.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Resize(self.config.transforms.image_size, self.config.transforms.image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        if stage == 'train' or stage==None:
            self.t_dataset = Dataset(
                self.paths['train_images'],
                self.paths['train_labels'],
                transform=train_transform
            )
        if stage == 'val' or stage==None:
            self.v_dataset = Dataset(
                self.paths['test_images'],
                self.paths['test_labels'],
                transform=val_transform
            )
    def train_dataloader(self):
        return DataLoader(
            self.t_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    def val_dataloader(self):
        return DataLoader(
            self.v_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    