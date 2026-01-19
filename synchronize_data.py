#!/usr/bin/env python3
"""
传感器数据时间同步脚本
将图像、点云、IMU等数据按时间戳对齐
"""

import json
import numpy as np
from pathlib import Path
import cv2
import pickle
from scipy.interpolate import interp1d


class DataSynchronizer:
    def __init__(self, data_dir, time_threshold_ns=50_000_000):  # 50ms
        """
        Args:
            data_dir: 提取数据的目录
            time_threshold_ns: 时间同步阈值(纳秒)，默认50ms
        """
        self.data_dir = Path(data_dir)
        self.time_threshold = time_threshold_ns
        self.load_metadata()

    def load_metadata(self):
        """加载时间戳索引和其他元数据"""
        print("加载元数据...")

        # 加载时间戳索引
        with open(self.data_dir / 'timestamp_index.json', 'r') as f:
            self.timestamp_index = json.load(f)

        # 加载IMU数据
        with open(self.data_dir / 'imu/imu_data.json', 'r') as f:
            self.imu_data = json.load(f)

        # 加载里程计数据
        with open(self.data_dir / 'odometry/odometry_data.json', 'r') as f:
            self.odometry_data = json.load(f)

        # 加载相机标定
        calib_path = self.data_dir / 'camera_info/calibration.json'
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                self.camera_info = json.load(f)
        else:
            self.camera_info = None

        print(f"✓ 左相机图像: {len(self.timestamp_index['left_images'])} 帧")
        print(f"✓ 右相机图像: {len(self.timestamp_index['right_images'])} 帧")
        print(f"✓ 点云数据: {len(self.timestamp_index['pointclouds'])} 帧")
        print(f"✓ IMU数据: {len(self.imu_data)} 条")
        print(f"✓ 里程计数据: {len(self.odometry_data)} 条")

    def find_nearest_timestamp(self, target_ts, data_list, ts_key='timestamp'):
        """找到最接近目标时间戳的数据"""
        if not data_list:
            return None, float('inf')

        timestamps = [d[ts_key] for d in data_list]
        idx = np.searchsorted(timestamps, target_ts)

        # 检查边界
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs(timestamps[idx - 1] - target_ts)))
        if idx < len(timestamps):
            candidates.append((idx, abs(timestamps[idx] - target_ts)))

        if not candidates:
            return None, float('inf')

        best_idx, best_diff = min(candidates, key=lambda x: x[1])

        if best_diff <= self.time_threshold:
            return data_list[best_idx], best_diff
        else:
            return None, best_diff

    def interpolate_imu(self, timestamp):
        """对IMU数据进行线性插值"""
        if len(self.imu_data) < 2:
            return None

        timestamps = np.array([d['timestamp'] for d in self.imu_data])

        # 检查时间戳范围
        if timestamp < timestamps[0] or timestamp > timestamps[-1]:
            return None

        # 提取需要插值的数据
        angular_vel_x = np.array([d['angular_velocity']['x'] for d in self.imu_data])
        angular_vel_y = np.array([d['angular_velocity']['y'] for d in self.imu_data])
        angular_vel_z = np.array([d['angular_velocity']['z'] for d in self.imu_data])

        linear_acc_x = np.array([d['linear_acceleration']['x'] for d in self.imu_data])
        linear_acc_y = np.array([d['linear_acceleration']['y'] for d in self.imu_data])
        linear_acc_z = np.array([d['linear_acceleration']['z'] for d in self.imu_data])

        # 创建插值函数
        interp_funcs = {
            'angular_velocity': {
                'x': interp1d(timestamps, angular_vel_x, kind='linear'),
                'y': interp1d(timestamps, angular_vel_y, kind='linear'),
                'z': interp1d(timestamps, angular_vel_z, kind='linear')
            },
            'linear_acceleration': {
                'x': interp1d(timestamps, linear_acc_x, kind='linear'),
                'y': interp1d(timestamps, linear_acc_y, kind='linear'),
                'z': interp1d(timestamps, linear_acc_z, kind='linear')
            }
        }

        # 执行插值
        result = {
            'timestamp': int(timestamp),
            'angular_velocity': {
                'x': float(interp_funcs['angular_velocity']['x'](timestamp)),
                'y': float(interp_funcs['angular_velocity']['y'](timestamp)),
                'z': float(interp_funcs['angular_velocity']['z'](timestamp))
            },
            'linear_acceleration': {
                'x': float(interp_funcs['linear_acceleration']['x'](timestamp)),
                'y': float(interp_funcs['linear_acceleration']['y'](timestamp)),
                'z': float(interp_funcs['linear_acceleration']['z'](timestamp))
            }
        }

        return result

    def synchronize_all(self, use_imu_interpolation=True):
        """同步所有传感器数据"""
        print(f"\n开始同步数据 (阈值: {self.time_threshold / 1e6:.1f}ms)...")

        synchronized_frames = []

        # 以左相机为基准进行同步
        for i, left_img_info in enumerate(self.timestamp_index['left_images']):
            target_ts = left_img_info['timestamp']

            frame_data = {
                'frame_id': i,
                'timestamp': target_ts,
                'left_image': left_img_info
            }

            # 同步右相机
            right_match, right_diff = self.find_nearest_timestamp(
                target_ts, self.timestamp_index['right_images']
            )
            if right_match:
                frame_data['right_image'] = right_match
                frame_data['stereo_time_diff_ms'] = right_diff / 1e6

            # 同步点云
            pc_match, pc_diff = self.find_nearest_timestamp(
                target_ts, self.timestamp_index['pointclouds']
            )
            if pc_match:
                frame_data['pointcloud'] = pc_match
                frame_data['pointcloud_time_diff_ms'] = pc_diff / 1e6

            # 同步IMU (插值或最近邻)
            if use_imu_interpolation:
                imu_data = self.interpolate_imu(target_ts)
                if imu_data:
                    frame_data['imu'] = imu_data
                    frame_data['imu_interpolated'] = True
            else:
                imu_match, imu_diff = self.find_nearest_timestamp(
                    target_ts, self.imu_data
                )
                if imu_match:
                    frame_data['imu'] = imu_match
                    frame_data['imu_time_diff_ms'] = imu_diff / 1e6
                    frame_data['imu_interpolated'] = False

            # 同步里程计
            odom_match, odom_diff = self.find_nearest_timestamp(
                target_ts, self.odometry_data
            )
            if odom_match:
                frame_data['odometry'] = odom_match
                frame_data['odometry_time_diff_ms'] = odom_diff / 1e6

            synchronized_frames.append(frame_data)

            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"已同步 {i + 1}/{len(self.timestamp_index['left_images'])} 帧...")

        # 保存同步结果
        self.save_synchronized_data(synchronized_frames)

        # 打印统计信息
        self.print_sync_statistics(synchronized_frames)

        return synchronized_frames

    def save_synchronized_data(self, synchronized_frames):
        """保存同步后的数据"""
        output_path = self.data_dir / 'synchronized/sync_data.json'

        with open(output_path, 'w') as f:
            json.dump(synchronized_frames, f, indent=2)

        print(f"\n✓ 同步数据已保存到: {output_path}")

        # 创建简化的CSV文件便于查看
        csv_path = self.data_dir / 'synchronized/sync_summary.csv'
        with open(csv_path, 'w') as f:
            f.write("frame_id,timestamp,has_left,has_right,has_pointcloud,has_imu,has_odometry\n")
            for frame in synchronized_frames:
                f.write(f"{frame['frame_id']},{frame['timestamp']},"
                        f"{1 if 'left_image' in frame else 0},"
                        f"{1 if 'right_image' in frame else 0},"
                        f"{1 if 'pointcloud' in frame else 0},"
                        f"{1 if 'imu' in frame else 0},"
                        f"{1 if 'odometry' in frame else 0}\n")

        print(f"✓ 同步摘要已保存到: {csv_path}")

    def print_sync_statistics(self, synchronized_frames):
        """打印同步统计信息"""
        total = len(synchronized_frames)

        stats = {
            'with_stereo': 0,
            'with_pointcloud': 0,
            'with_imu': 0,
            'with_odometry': 0,
            'complete_frames': 0  # 包含所有传感器
        }

        for frame in synchronized_frames:
            has_stereo = 'left_image' in frame and 'right_image' in frame
            has_pc = 'pointcloud' in frame
            has_imu = 'imu' in frame
            has_odom = 'odometry' in frame

            if has_stereo:
                stats['with_stereo'] += 1
            if has_pc:
                stats['with_pointcloud'] += 1
            if has_imu:
                stats['with_imu'] += 1
            if has_odom:
                stats['with_odometry'] += 1
            if has_stereo and has_pc and has_imu and has_odom:
                stats['complete_frames'] += 1

        print("\n" + "=" * 60)
        print("同步统计:")
        print("=" * 60)
        print(f"总帧数: {total}")
        print(f"包含立体图像对: {stats['with_stereo']} ({stats['with_stereo'] / total * 100:.1f}%)")
        print(f"包含点云: {stats['with_pointcloud']} ({stats['with_pointcloud'] / total * 100:.1f}%)")
        print(f"包含IMU: {stats['with_imu']} ({stats['with_imu'] / total * 100:.1f}%)")
        print(f"包含里程计: {stats['with_odometry']} ({stats['with_odometry'] / total * 100:.1f}%)")
        print(f"完整帧(所有传感器): {stats['complete_frames']} ({stats['complete_frames'] / total * 100:.1f}%)")
        print("=" * 60)

    def create_annotation_subset(self, num_frames=200, output_dir=None):
        """创建用于标注的子集"""
        if output_dir is None:
            output_dir = self.data_dir / 'annotation_subset'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载同步数据
        with open(self.data_dir / 'synchronized/sync_data.json', 'r') as f:
            sync_data = json.load(f)

        # 筛选完整帧
        complete_frames = [
            f for f in sync_data
            if 'left_image' in f and 'right_image' in f and 'pointcloud' in f
        ]

        # 均匀采样
        if len(complete_frames) > num_frames:
            indices = np.linspace(0, len(complete_frames) - 1, num_frames, dtype=int)
            selected_frames = [complete_frames[i] for i in indices]
        else:
            selected_frames = complete_frames

        print(f"\n创建标注子集: {len(selected_frames)} 帧")

        # 复制选中的图像
        for i, frame in enumerate(selected_frames):
            # 左图像
            src_left = self.data_dir / 'images/left' / frame['left_image']['filename']
            dst_left = output_dir / f'frame_{i:04d}_left.jpg'
            img = cv2.imread(str(src_left))
            cv2.imwrite(str(dst_left), img)

            # 右图像
            if 'right_image' in frame:
                src_right = self.data_dir / 'images/right' / frame['right_image']['filename']
                dst_right = output_dir / f'frame_{i:04d}_right.jpg'
                img = cv2.imread(str(src_right))
                cv2.imwrite(str(dst_right), img)

        # 保存子集元数据
        subset_metadata = {
            'num_frames': len(selected_frames),
            'frames': selected_frames
        }
        with open(output_dir / 'subset_metadata.json', 'w') as f:
            json.dump(subset_metadata, f, indent=2)

        print(f"✓ 标注子集已保存到: {output_dir}")
        print(f"  - 包含 {len(selected_frames)} 组立体图像对")

        return selected_frames, output_dir


if __name__ == '__main__':
    # 配置路径
    DATA_DIR = './extracted_data'

    # 创建同步器
    synchronizer = DataSynchronizer(DATA_DIR, time_threshold_ns=50_000_000)

    # 同步所有数据
    synchronized_frames = synchronizer.synchronize_all(use_imu_interpolation=True)

    # 创建标注子集 (100-200帧)
    subset_frames, subset_dir = synchronizer.create_annotation_subset(num_frames=150)

    print("\n✓ 数据同步完成!")