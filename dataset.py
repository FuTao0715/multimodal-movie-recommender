"""
dataset.py - 多任务数据集加载类（修复版）
正确读取MVTec AD格式的掩码
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MultiTaskDataset(Dataset):
    """
    多任务数据集：同时支持分类和分割任务
    分类标签：0=螺母(nut), 1=螺丝(screw)
    分割标签：0=背景, 1=缺陷
    """
    def __init__(self, data_dir, image_size=(416, 416), is_train=True, use_cutmix=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        self.use_cutmix = use_cutmix and is_train

        split = 'train' if is_train else 'val'
        self.images_dir = os.path.join(data_dir, split, 'images')
        self.masks_dir = os.path.join(data_dir, split, 'masks')

        self.image_paths = sorted(glob(os.path.join(self.images_dir, '*.png')))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        print(f"Loaded {len(self.image_paths)} samples for {split} split")

        self.transform = self._build_transform()

    def _build_transform(self):
        """构建数据增强pipeline"""
        if self.is_train:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码 - 关键修复
        mask_path = os.path.join(self.masks_dir, filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Cannot read mask {mask_path}, creating empty mask")
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                # 确保掩码是二值：255->1, 其他->0
                mask = (mask > 127).astype(np.uint8)
        else:
            # 如果掩码不存在，创建全黑掩码（无缺陷）
            print(f"Warning: Mask not found {mask_path}, creating empty mask")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 获取分类标签：从文件名前缀获取
        if filename.startswith('nut_'):
            label = 0  # 螺母
        elif filename.startswith('screw_'):
            label = 1  # 螺丝
        else:
            # 根据文件名规则推断
            if 'screw' in filename.lower():
                label = 1
            else:
                label = 0

        # 应用频域滤波
        if self.is_train and random.random() < 0.3:
            image = self._apply_frequency_filter(image)

        # 应用直方图均衡化
        if self.is_train and random.random() < 0.3:
            image = self._apply_histogram_equalization(image)

        # 应用数据增强
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image'].float() / 255.0
        mask = transformed['mask'].long()

        # CutMix增强
        if self.is_train and self.use_cutmix and random.random() < 0.5:
            idx2 = random.randint(0, len(self.image_paths) - 1)
            while idx2 == idx:
                idx2 = random.randint(0, len(self.image_paths) - 1)

            image2, mask2, label2, _ = self._get_item_without_aug(idx2)
            image, mask, label = self._apply_cutmix(image, mask, label, image2, mask2, label2)

        return {
            'image': image,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long),
            'filename': filename
        }

    def _get_item_without_aug(self, idx):
        """获取未增强的样本（用于CutMix）"""
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.masks_dir, filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 调整尺寸
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        if filename.startswith('nut_'):
            label = 0
        elif filename.startswith('screw_'):
            label = 1
        else:
            label = 0

        return image, mask, label, filename

    def _apply_cutmix(self, img1, mask1, label1, img2, mask2, label2):
        _, H, W = img1.shape

        cut_ratio = random.uniform(0.2, 0.5)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)

        cx = random.randint(0, W - cut_w)
        cy = random.randint(0, H - cut_h)

        img_mixed = img1.clone()
        img_mixed[:, cy:cy+cut_h, cx:cx+cut_w] = img2[:, cy:cy+cut_h, cx:cx+cut_w]

        mask_mixed = mask1.clone()
        mask_mixed[cy:cy+cut_h, cx:cx+cut_w] = mask2[cy:cy+cut_h, cx:cx+cut_w]

        lambda_val = (cut_w * cut_h) / (W * H)
        label_mixed = label1 * (1 - lambda_val) + label2 * lambda_val
        label_mixed = torch.round(torch.tensor(label_mixed, dtype=torch.float32)).long()

        return img_mixed, mask_mixed, label_mixed

    def _apply_frequency_filter(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        mask_hp = np.ones((rows, cols), np.uint8)
        r = 30
        mask_hp[crow-r:crow+r, ccol-r:ccol+r] = 0

        fshift_filtered = fshift * mask_hp
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced = cv2.addWeighted(gray, 0.7, img_back, 0.3, 0)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return result

    def _apply_histogram_equalization(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return result


def create_dataloaders(data_dir, batch_size=16, image_size=(416, 416), num_workers=4):
    train_dataset = MultiTaskDataset(data_dir, image_size, is_train=True, use_cutmix=True)
    val_dataset = MultiTaskDataset(data_dir, image_size, is_train=False, use_cutmix=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


if __name__ == "__main__":
    data_dir = "./dataset/metal_fastener_dataset"

    if not os.path.exists(data_dir):
        data_dir = "../dataset/metal_fastener_dataset"
        if not os.path.exists(data_dir):
            print(f"Dataset directory not found!")
            exit(1)

    try:
        train_loader, val_loader, train_size, val_size = create_dataloaders(
            data_dir, batch_size=4, image_size=(416, 416)
        )

        print(f"Train samples: {train_size}, Val samples: {val_size}")

        for batch in train_loader:
            print(f"Image shape: {batch['image'].shape}")
            print(f"Mask shape: {batch['mask'].shape}")
            print(f"Label shape: {batch['label'].shape}")
            print(f"Labels: {batch['label']}")
            print(f"Filenames: {batch['filename']}")

            labels = batch['label'].numpy()
            print(f"Label distribution - Nut (0): {sum(labels == 0)}, Screw (1): {sum(labels == 1)}")

            masks = batch['mask'].numpy()
            print(f"Mask values - min: {masks.min()}, max: {masks.max()}")
            print(f"Number of defective pixels in batch: {np.sum(masks > 0)}")
            break

    except Exception as e:
        print(f"Error loading dataset: {e}")