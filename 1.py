# list_mask_files.py - 列出实际的掩码文件
import os

screw_gt = "F:/机器学习/数据集/screw/ground_truth"
if os.path.exists(screw_gt):
    for defect_type in os.listdir(screw_gt):
        defect_dir = os.path.join(screw_gt, defect_type)
        if os.path.isdir(defect_dir):
            files = os.listdir(defect_dir)
            print(f"Screw/{defect_type}: {files[:5]}")

nut_gt = "F:/机器学习/数据集/metal_nut/ground_truth"
if os.path.exists(nut_gt):
    for defect_type in os.listdir(nut_gt):
        defect_dir = os.path.join(nut_gt, defect_type)
        if os.path.isdir(defect_dir):
            files = os.listdir(defect_dir)
            print(f"Nut/{defect_type}: {files[:5]}")