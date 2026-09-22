"""
preprocess_data.py - 数据集预处理脚本（修复版）
正确处理 MVTec AD 数据集的掩码（文件名带 _mask 后缀）
"""

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


class MVTecDataPreprocessor:
    """MVTec AD数据集预处理器"""

    def __init__(self, root_dir, output_dir="./dataset/metal_fastener_dataset"):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.screw_dir = self.root_dir / "screw"
        self.nut_dir = self.root_dir / "metal_nut"

        # 缺陷类别映射
        self.screw_defect_types = [
            'manipulated_front',
            'scratch_head',
            'scratch_neck',
            'thread_side',
            'thread_top'
        ]

        self.nut_defect_types = [
            'bent',
            'color',
            'flip',
            'scratch'
        ]

        self.all_samples = []

    def scan_samples(self):
        """扫描所有样本"""
        print("Scanning samples...")
        self._scan_screw_samples()
        self._scan_nut_samples()
        print(f"Found {len(self.all_samples)} total samples")
        return self.all_samples

    def _scan_screw_samples(self):
        """扫描螺丝样本"""
        # 训练集
        train_dir = self.screw_dir / "train"
        for defect_type in ['good'] + self.screw_defect_types:
            defect_dir = train_dir / defect_type
            if not defect_dir.exists():
                print(f"Warning: {defect_dir} not found")
                continue

            for img_file in defect_dir.glob("*.png"):
                sample = {
                    'source': 'screw',
                    'prefix': 'screw',
                    'defect_type': defect_type,
                    'is_good': defect_type == 'good',
                    'image_path': str(img_file),
                    'mask_path': None
                }

                if not sample['is_good']:
                    # 掩码文件名是 000_mask.png 格式
                    mask_dir = self.screw_dir / "ground_truth" / defect_type
                    mask_name = img_file.stem + "_mask.png"  # 添加 _mask 后缀
                    mask_file = mask_dir / mask_name
                    if mask_file.exists():
                        sample['mask_path'] = str(mask_file)
                    else:
                        print(f"Warning: Mask not found for {img_file} (expected {mask_name})")

                self.all_samples.append(sample)

        # 测试集
        test_dir = self.screw_dir / "test"
        for defect_type in ['good'] + self.screw_defect_types:
            defect_dir = test_dir / defect_type
            if not defect_dir.exists():
                continue

            for img_file in defect_dir.glob("*.png"):
                sample = {
                    'source': 'screw',
                    'prefix': 'screw',
                    'defect_type': defect_type,
                    'is_good': defect_type == 'good',
                    'image_path': str(img_file),
                    'mask_path': None
                }

                if not sample['is_good']:
                    mask_dir = self.screw_dir / "ground_truth" / defect_type
                    mask_name = img_file.stem + "_mask.png"
                    mask_file = mask_dir / mask_name
                    if mask_file.exists():
                        sample['mask_path'] = str(mask_file)

                self.all_samples.append(sample)

    def _scan_nut_samples(self):
        """扫描螺母样本"""
        # 训练集
        train_dir = self.nut_dir / "train"
        for defect_type in ['good'] + self.nut_defect_types:
            defect_dir = train_dir / defect_type
            if not defect_dir.exists():
                print(f"Warning: {defect_dir} not found")
                continue

            for img_file in defect_dir.glob("*.png"):
                sample = {
                    'source': 'nut',
                    'prefix': 'nut',
                    'defect_type': defect_type,
                    'is_good': defect_type == 'good',
                    'image_path': str(img_file),
                    'mask_path': None
                }

                if not sample['is_good']:
                    mask_dir = self.nut_dir / "ground_truth" / defect_type
                    mask_name = img_file.stem + "_mask.png"
                    mask_file = mask_dir / mask_name
                    if mask_file.exists():
                        sample['mask_path'] = str(mask_file)
                    else:
                        print(f"Warning: Mask not found for {img_file} (expected {mask_name})")

                self.all_samples.append(sample)

        # 测试集
        test_dir = self.nut_dir / "test"
        for defect_type in ['good'] + self.nut_defect_types:
            defect_dir = test_dir / defect_type
            if not defect_dir.exists():
                continue

            for img_file in defect_dir.glob("*.png"):
                sample = {
                    'source': 'nut',
                    'prefix': 'nut',
                    'defect_type': defect_type,
                    'is_good': defect_type == 'good',
                    'image_path': str(img_file),
                    'mask_path': None
                }

                if not sample['is_good']:
                    mask_dir = self.nut_dir / "ground_truth" / defect_type
                    mask_name = img_file.stem + "_mask.png"
                    mask_file = mask_dir / mask_name
                    if mask_file.exists():
                        sample['mask_path'] = str(mask_file)

                self.all_samples.append(sample)

    def prepare_output_dirs(self):
        """创建输出目录"""
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

        print(f"Output directory created at: {self.output_dir}")

    def process_samples(self, train_ratio=0.7, seed=42):
        random.seed(seed)

        self.prepare_output_dirs()
        random.shuffle(self.all_samples)

        split_idx = int(len(self.all_samples) * train_ratio)
        train_samples = self.all_samples[:split_idx]
        val_samples = self.all_samples[split_idx:]

        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")

        # 处理训练集
        print("\nProcessing training set...")
        for sample in tqdm(train_samples):
            self._process_single_sample(sample, 'train')

        # 处理验证集
        print("\nProcessing validation set...")
        for sample in tqdm(val_samples):
            self._process_single_sample(sample, 'val')

        print("\nDataset preprocessing completed!")
        self._generate_report(train_samples, val_samples)

    def _process_single_sample(self, sample, split):
        """处理单个样本"""
        img_path = Path(sample['image_path'])
        new_name = f"{sample['prefix']}_{img_path.name}"

        dst_img_dir = self.output_dir / split / 'images'
        dst_mask_dir = self.output_dir / split / 'masks'

        dst_img_path = dst_img_dir / new_name
        dst_mask_path = dst_mask_dir / new_name

        # 复制图像
        shutil.copy2(sample['image_path'], dst_img_path)

        # 处理掩码
        if sample['is_good']:
            # 良品：生成全黑掩码
            img = cv2.imread(sample['image_path'])
            if img is not None:
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(str(dst_mask_path), mask)
        else:
            # 缺陷品：复制掩码
            if sample['mask_path'] and Path(sample['mask_path']).exists():
                # 直接复制掩码文件
                shutil.copy2(sample['mask_path'], dst_mask_path)

                # 验证掩码是否正确复制
                verify_mask = cv2.imread(str(dst_mask_path), cv2.IMREAD_GRAYSCALE)
                if verify_mask is not None:
                    unique_vals = np.unique(verify_mask)
                    if len(unique_vals) > 1:
                        print(f"  ✓ Mask verified for {new_name}: values {unique_vals}")
                    else:
                        print(f"  ⚠ Warning: Mask has only value {unique_vals} for {new_name}")
            else:
                # 如果没有掩码，创建全黑掩码
                print(f"  ⚠ No mask for {sample['image_path']}, creating empty mask")
                img = cv2.imread(sample['image_path'])
                if img is not None:
                    h, w = img.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.imwrite(str(dst_mask_path), mask)

    def _generate_report(self, train_samples, val_samples):
        """生成数据集统计报告"""
        report_path = self.output_dir / "dataset_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Dataset Preprocessing Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total samples: {len(self.all_samples)}\n")
            f.write(f"Train samples: {len(train_samples)}\n")
            f.write(f"Val samples: {len(val_samples)}\n\n")

            # 按类别统计
            f.write("--- By Category ---\n")
            screw_count = sum(1 for s in self.all_samples if s['source'] == 'screw')
            nut_count = sum(1 for s in self.all_samples if s['source'] == 'nut')
            f.write(f"Screw: {screw_count}\n")
            f.write(f"Nut: {nut_count}\n\n")

            # 训练集缺陷分布
            train_defect_counts = {}
            for s in train_samples:
                if not s['is_good']:
                    key = f"{s['source']}_{s['defect_type']}"
                    train_defect_counts[key] = train_defect_counts.get(key, 0) + 1

            f.write("--- Train Set Defect Distribution ---\n")
            for defect, count in sorted(train_defect_counts.items()):
                f.write(f"  {defect}: {count}\n")

            f.write("\n--- Val Set Defect Distribution ---\n")
            val_defect_counts = {}
            for s in val_samples:
                if not s['is_good']:
                    key = f"{s['source']}_{s['defect_type']}"
                    val_defect_counts[key] = val_defect_counts.get(key, 0) + 1

            for defect, count in sorted(val_defect_counts.items()):
                f.write(f"  {defect}: {count}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess MVTec AD dataset')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory containing screw and metal_nut folders')
    parser.add_argument('--output_dir', type=str, default='./dataset/metal_fastener_dataset',
                        help='Output directory for processed dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train/val split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    preprocessor = MVTecDataPreprocessor(args.root_dir, args.output_dir)
    preprocessor.scan_samples()
    preprocessor.process_samples(train_ratio=args.train_ratio, seed=args.seed)


if __name__ == "__main__":
    main()