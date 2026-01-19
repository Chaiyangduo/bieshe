#!/usr/bin/env python3
"""
处理CVAT导出的标注数据
支持COCO和YOLO格式转换
"""

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict


class CVATAnnotationProcessor:
    def __init__(self, annotation_dir, image_dir, output_dir):
        self.annotation_dir = Path(annotation_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.setup_output_dirs()

    def setup_output_dirs(self):
        """创建输出目录"""
        dirs = [
            'images/train',
            'images/val',
            'labels/train',
            'labels/val',
            'visualizations',
            'coco_format'
        ]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

    def process_coco_annotations(self, split_ratio=0.8):
        """处理COCO格式的标注"""
        print("处理COCO格式标注...")

        # 读取COCO标注文件
        coco_file = self.annotation_dir / 'instances_default.json'
        if not coco_file.exists():
            print(f"错误: 找不到COCO标注文件 {coco_file}")
            return

        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        print(f"✓ 找到 {len(coco_data['images'])} 张图像")
        print(f"✓ 找到 {len(coco_data['annotations'])} 个标注")
        print(f"✓ 类别数: {len(coco_data['categories'])}")

        # 打印类别信息
        print("\n类别列表:")
        for cat in coco_data['categories']:
            print(f"  ID {cat['id']}: {cat['name']}")

        # 分割训练集和验证集
        num_images = len(coco_data['images'])
        num_train = int(num_images * split_ratio)

        indices = np.random.permutation(num_images)
        train_indices = set(indices[:num_train].tolist())
        val_indices = set(indices[num_train:].tolist())

        # 创建训练集和验证集的COCO文件
        train_coco = self.create_split_coco(coco_data, train_indices, 'train')
        val_coco = self.create_split_coco(coco_data, val_indices, 'val')

        # 保存分割后的COCO文件
        with open(self.output_dir / 'coco_format/train.json', 'w') as f:
            json.dump(train_coco, f, indent=2)

        with open(self.output_dir / 'coco_format/val.json', 'w') as f:
            json.dump(val_coco, f, indent=2)

        print(f"\n✓ 训练集: {len(train_coco['images'])} 张图像")
        print(f"✓ 验证集: {len(val_coco['images'])} 张图像")

        # 复制图像文件
        self.copy_images(coco_data, train_indices, val_indices)

        # 转换为YOLO格式
        self.convert_coco_to_yolo(train_coco, 'train')
        self.convert_coco_to_yolo(val_coco, 'val')

        # 创建数据集配置文件
        self.create_dataset_yaml(coco_data['categories'])

        return train_coco, val_coco

    def create_split_coco(self, coco_data, image_indices, split_name):
        """创建训练/验证集的COCO格式数据"""
        # 获取该分割的图像
        split_images = [img for i, img in enumerate(coco_data['images'])
                        if i in image_indices]
        split_image_ids = set(img['id'] for img in split_images)

        # 获取对应的标注
        split_annotations = [ann for ann in coco_data['annotations']
                             if ann['image_id'] in split_image_ids]

        # 创建新的COCO字典
        split_coco = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data['categories'],
            'images': split_images,
            'annotations': split_annotations
        }

        return split_coco

    def copy_images(self, coco_data, train_indices, val_indices):
        """复制图像到训练/验证集目录"""
        print("\n复制图像文件...")

        for i, img_info in enumerate(coco_data['images']):
            src_path = self.image_dir / img_info['file_name']

            if not src_path.exists():
                print(f"警告: 图像文件不存在 {src_path}")
                continue

            # 确定目标目录
            if i in train_indices:
                dst_path = self.output_dir / 'images/train' / img_info['file_name']
            else:
                dst_path = self.output_dir / 'images/val' / img_info['file_name']

            shutil.copy2(src_path, dst_path)

        print(f"✓ 图像复制完成")

    def convert_coco_to_yolo(self, coco_data, split):
        """将COCO格式转换为YOLO格式"""
        print(f"\n转换 {split} 集为YOLO格式...")

        # 创建图像ID到标注的映射
        img_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            img_to_anns[ann['image_id']].append(ann)

        # 创建类别ID映射 (从1开始转换为从0开始)
        category_id_map = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}

        for img_info in coco_data['images']:
            img_id = img_info['id']
            img_w = img_info['width']
            img_h = img_info['height']

            # YOLO标注文件名
            label_filename = Path(img_info['file_name']).stem + '.txt'
            label_path = self.output_dir / f'labels/{split}' / label_filename

            # 转换该图像的所有标注
            yolo_annotations = []
            for ann in img_to_anns[img_id]:
                # COCO bbox格式: [x, y, width, height]
                x, y, w, h = ann['bbox']

                # 转换为YOLO格式: [class_id, x_center, y_center, width, height] (归一化)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                class_id = category_id_map[ann['category_id']]

                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                )

            # 保存YOLO标注文件
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

        print(f"✓ {split} 集YOLO标注转换完成")

    def create_dataset_yaml(self, categories):
        """创建YOLO数据集配置文件"""
        # 提取类别名称
        class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]

        yaml_content = f"""# 水下AUV数据集配置
# 用于YOLO训练

# 数据集路径
path: {self.output_dir.absolute()}
train: images/train
val: images/val

# 类别数量
nc: {len(class_names)}

# 类别名称
names: {class_names}

# 数据集信息
info:
  description: "Underwater AUV Dataset with stereo cameras and sonar"
  sensors: "Stereo camera, 3D sonar, IMU, depth sensor"
  duration_seconds: 148.6
  total_frames: {len(class_names)}
"""

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"\n✓ 数据集配置文件已创建: {yaml_path}")

    def visualize_annotations(self, coco_data, split, num_samples=10):
        """可视化标注"""
        print(f"\n生成 {split} 集标注可视化...")

        # 随机选择样本
        sample_images = np.random.choice(
            coco_data['images'],
            min(num_samples, len(coco_data['images'])),
            replace=False
        )

        # 创建类别颜色映射
        np.random.seed(42)
        colors = {}
        for cat in coco_data['categories']:
            colors[cat['id']] = tuple(np.random.randint(0, 255, 3).tolist())

        # 创建图像ID到标注的映射
        img_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            img_to_anns[ann['image_id']].append(ann)

        for img_info in sample_images:
            # 读取图像
            img_path = self.output_dir / f'images/{split}' / img_info['file_name']
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))

            # 绘制标注
            for ann in img_to_anns[img_info['id']]:
                x, y, w, h = map(int, ann['bbox'])
                cat_id = ann['category_id']
                color = colors[cat_id]

                # 绘制边界框
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # 添加标签
                cat_name = next(c['name'] for c in coco_data['categories']
                                if c['id'] == cat_id)
                label = f"{cat_name}"

                # 绘制标签背景
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(img, (x, y - label_h - 5), (x + label_w, y), color, -1)
                cv2.putText(img, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 保存可视化结果
            vis_filename = f"{split}_{Path(img_info['file_name']).stem}_vis.jpg"
            vis_path = self.output_dir / 'visualizations' / vis_filename
            cv2.imwrite(str(vis_path), img)

        print(f"✓ 可视化完成: {self.output_dir / 'visualizations'}")

    def generate_statistics(self, train_coco, val_coco):
        """生成数据集统计信息"""
        print("\n" + "=" * 60)
        print("数据集统计")
        print("=" * 60)

        # 计算每个类别的标注数量
        train_cat_counts = defaultdict(int)
        val_cat_counts = defaultdict(int)

        for ann in train_coco['annotations']:
            train_cat_counts[ann['category_id']] += 1

        for ann in val_coco['annotations']:
            val_cat_counts[ann['category_id']] += 1

        # 打印统计
        print(f"\n训练集:")
        print(f"  图像数量: {len(train_coco['images'])}")
        print(f"  标注数量: {len(train_coco['annotations'])}")
        print(f"  平均每张图像标注数: {len(train_coco['annotations']) / len(train_coco['images']):.2f}")

        print(f"\n验证集:")
        print(f"  图像数量: {len(val_coco['images'])}")
        print(f"  标注数量: {len(val_coco['annotations'])}")
        print(f"  平均每张图像标注数: {len(val_coco['annotations']) / len(val_coco['images']):.2f}")

        print(f"\n类别分布:")
        print(f"{'类别':<20} {'训练集':<10} {'验证集':<10} {'总计':<10}")
        print("-" * 50)

        for cat in train_coco['categories']:
            cat_id = cat['id']
            cat_name = cat['name']
            train_count = train_cat_counts[cat_id]
            val_count = val_cat_counts[cat_id]
            total = train_count + val_count
            print(f"{cat_name:<20} {train_count:<10} {val_count:<10} {total:<10}")

        print("=" * 60)

        # 保存统计到文件
        stats = {
            'train': {
                'num_images': len(train_coco['images']),
                'num_annotations': len(train_coco['annotations']),
                'category_counts': dict(train_cat_counts)
            },
            'val': {
                'num_images': len(val_coco['images']),
                'num_annotations': len(val_coco['annotations']),
                'category_counts': dict(val_cat_counts)
            },
            'categories': {cat['id']: cat['name'] for cat in train_coco['categories']}
        }

        with open(self.output_dir / 'dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='处理CVAT导出的标注数据')
    parser.add_argument('--annotation_dir', type=str, required=True,
                        help='CVAT导出的标注目录')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='图像目录')
    parser.add_argument('--output_dir', type=str, default='./processed_dataset',
                        help='输出目录')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--visualize', action='store_true',
                        help='生成标注可视化')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                        help='可视化样本数量')

    args = parser.parse_args()

    # 创建处理器
    processor = CVATAnnotationProcessor(
        args.annotation_dir,
        args.image_dir,
        args.output_dir
    )

    # 处理标注
    train_coco, val_coco = processor.process_coco_annotations(args.split_ratio)

    # 生成统计
    processor.generate_statistics(train_coco, val_coco)

    # 可视化
    if args.visualize:
        processor.visualize_annotations(train_coco, 'train', args.num_vis_samples)
        processor.visualize_annotations(val_coco, 'val', args.num_vis_samples)

    print("\n✓ 所有处理完成!")
    print(f"✓ 输出目录: {args.output_dir}")