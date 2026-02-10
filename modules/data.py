import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

def split_data(images_dir, labels_dir, output_dir, test_size=0.2):
    for split in ['train', 'test']:
        for dtype in ['images', 'labels']:
            (Path(output_dir) / split / dtype).mkdir(parents=True, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('_label.png')]
    patient_files = {}
    for lf in label_files:
        pid = lf.split('_')[0]
        if pid not in patient_files:
            patient_files[pid] = []
        patient_files[pid].append(lf)
    
    patients = list(patient_files.keys())
    train_patients, test_patients = train_test_split(patients, test_size=test_size)
    def copy_group(patients_list, split_name):
        for pid in patients_list:
            for lf in patient_files[pid]:
                base = lf.replace('_label.png', '')
                img_file = f"{base}_image.png"
                src_img = os.path.join(images_dir, img_file)
                
                if os.path.exists(src_img):
                    shutil.copy2(
                        os.path.join(labels_dir, lf),
                        os.path.join(output_dir, split_name, 'labels', lf)
                    )
                    shutil.copy2(
                        src_img,
                        os.path.join(output_dir, split_name, 'images', img_file)
                    )
    copy_group(train_patients, 'train')
    copy_group(test_patients, 'test')
    
    return {
        'train_images': str(Path(output_dir) / 'train' / 'images'),
        'train_labels': str(Path(output_dir) / 'train' / 'labels'),
        'test_images': str(Path(output_dir) / 'test' / 'images'),
        'test_labels': str(Path(output_dir) / 'test' / 'labels'),
        'train_patients': train_patients,
        'test_patients': test_patients
    }
def split():
    from modules.config import config
    cfg = config()
    path = '/root/.cache/kagglehub/datasets/zaryabahmadkhan/2d-slicing-of-imagetbad-dataset/versions/1'
    im_path = os.path.join(path, '2D dataset', 'images')
    lb_path = os.path.join(path, '2D dataset', 'labels')
    result = split_data(
        images_dir=im_path,
        labels_dir=lb_path,
        output_dir=cfg.data.output,
        test_size=cfg.data.test_size
    )
    return {
        'train_images': result['train_images'],
        'train_labels': result['train_labels'],
        'test_images': result['test_images'],
        'test_labels': result['test_labels']
    }

class Dataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith('_image.png')])
        self.label_files = sorted([f for f in os.listdir(labels_dir) 
                                   if f.endswith('_label.png')])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        label = (label > 0).long()
        return image, label

def transformers():
    from modules.config import config
    cfg = config()
    
    train_transform = A.Compose([
        A.Resize(height=cfg.transforms.image_size, width=cfg.transforms.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=cfg.transforms.image_size, width=cfg.transforms.image_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform