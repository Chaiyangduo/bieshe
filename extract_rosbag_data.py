#!/usr/bin/env python3
"""
ROS2 Bag数据提取脚本 - 最终工作版本
直接使用 typestore.deserialize_cdr()
"""

import os
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
import json
import pickle


class RosbagExtractor:
    def __init__(self, bag_path, output_dir):
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        # 获取typestore
        self.typestore = get_typestore(Stores.LATEST)
        self.setup_output_dirs()

    def setup_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            'images/left',
            'images/right',
            'pointcloud',
            'sonar/range',
            'sonar/intensity',
            'imu',
            'depth',
            'odometry',
            'tf',
            'camera_info',
            'synchronized'
        ]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)
        print(f"✓ 输出目录创建完成: {self.output_dir}")

    @staticmethod
    def safe_bytes_convert(data):
        """安全地转换数据为bytes"""
        if isinstance(data, (bytes, bytearray)):
            return data
        elif hasattr(data, '__iter__'):
            try:
                return bytes(data)
            except:
                return bytes(list(data))
        else:
            return data

    def extract_compressed_image(self, msg, timestamp, topic_name):
        """提取压缩图像"""
        try:
            data_bytes = self.safe_bytes_convert(msg.data)
            image_data = np.frombuffer(data_bytes, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if image is not None:
                return {
                    'image': image,
                    'timestamp': timestamp,
                    'frame_id': msg.header.frame_id,
                    'format': msg.format
                }
        except Exception as e:
            print(f"图像解码错误 {topic_name}: {e}")
        return None

    def extract_pointcloud(self, msg, timestamp):
        """提取点云数据"""
        try:
            data_bytes = self.safe_bytes_convert(msg.data)

            pc_data = {
                'timestamp': timestamp,
                'frame_id': msg.header.frame_id,
                'height': msg.height,
                'width': msg.width,
                'fields': [(f.name, f.offset, f.datatype) for f in msg.fields],
                'point_step': msg.point_step,
                'row_step': msg.row_step,
                'data': np.frombuffer(data_bytes, dtype=np.uint8),
                'is_dense': msg.is_dense
            }
            return pc_data
        except Exception as e:
            print(f"点云提取错误: {e}")
        return None

    @staticmethod
    def extract_imu(msg, timestamp):
        """提取IMU数据"""
        return {
            'timestamp': timestamp,
            'frame_id': msg.header.frame_id,
            'orientation': {
                'x': float(msg.orientation.x),
                'y': float(msg.orientation.y),
                'z': float(msg.orientation.z),
                'w': float(msg.orientation.w)
            },
            'angular_velocity': {
                'x': float(msg.angular_velocity.x),
                'y': float(msg.angular_velocity.y),
                'z': float(msg.angular_velocity.z)
            },
            'linear_acceleration': {
                'x': float(msg.linear_acceleration.x),
                'y': float(msg.linear_acceleration.y),
                'z': float(msg.linear_acceleration.z)
            }
        }

    @staticmethod
    def extract_odometry(msg, timestamp):
        """提取里程计数据"""
        return {
            'timestamp': timestamp,
            'frame_id': msg.header.frame_id,
            'child_frame_id': msg.child_frame_id,
            'position': {
                'x': float(msg.pose.pose.position.x),
                'y': float(msg.pose.pose.position.y),
                'z': float(msg.pose.pose.position.z)
            },
            'orientation': {
                'x': float(msg.pose.pose.orientation.x),
                'y': float(msg.pose.pose.orientation.y),
                'z': float(msg.pose.pose.orientation.z),
                'w': float(msg.pose.pose.orientation.w)
            },
            'linear_velocity': {
                'x': float(msg.twist.twist.linear.x),
                'y': float(msg.twist.twist.linear.y),
                'z': float(msg.twist.twist.linear.z)
            },
            'angular_velocity': {
                'x': float(msg.twist.twist.angular.x),
                'y': float(msg.twist.twist.angular.y),
                'z': float(msg.twist.twist.angular.z)
            }
        }

    @staticmethod
    def extract_camera_info(msg, timestamp):
        """提取相机标定信息"""
        return {
            'timestamp': timestamp,
            'frame_id': msg.header.frame_id,
            'height': msg.height,
            'width': msg.width,
            'distortion_model': msg.distortion_model,
            'D': [float(x) for x in msg.d],
            'K': [float(x) for x in msg.k],
            'R': [float(x) for x in msg.r],
            'P': [float(x) for x in msg.p]
        }

    def extract_all(self):
        """提取所有数据"""
        print(f"\n开始提取bag文件: {self.bag_path}")

        counters = {
            'left_images': 0,
            'right_images': 0,
            'pointclouds': 0,
            'sonar_range': 0,
            'sonar_intensity': 0,
            'imu': 0,
            'depth': 0,
            'odometry': 0,
            'camera_info_left': 0,
            'camera_info_right': 0
        }

        all_data = {
            'left_images': [],
            'right_images': [],
            'pointclouds': [],
            'imu': [],
            'odometry': [],
            'camera_info': {}
        }

        try:
            with Reader(self.bag_path) as reader:
                connections = [c for c in reader.connections]

                print(f"\n找到 {len(connections)} 个topic")
                for conn in connections:
                    print(f"  - {conn.topic} ({conn.msgtype})")

                print("\n开始处理消息...")

                for connection, timestamp, rawdata in reader.messages():
                    try:
                        # 使用typestore的deserialize_cdr方法
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)

                        if connection.topic == '/auip/camera/forward/left/image/compressed':
                            img_data = self.extract_compressed_image(msg, timestamp, 'left')
                            if img_data:
                                filename = f"{counters['left_images']:06d}_{timestamp}.jpg"
                                filepath = self.output_dir / 'images/left' / filename
                                cv2.imwrite(str(filepath), img_data['image'])
                                all_data['left_images'].append({
                                    'timestamp': timestamp,
                                    'filename': filename,
                                    'frame_id': img_data['frame_id']
                                })
                                counters['left_images'] += 1
                                if counters['left_images'] == 1:
                                    print("✓ 成功提取第一张左相机图像!")

                        elif connection.topic == '/auip/camera/forward/right/image/compressed':
                            img_data = self.extract_compressed_image(msg, timestamp, 'right')
                            if img_data:
                                filename = f"{counters['right_images']:06d}_{timestamp}.jpg"
                                filepath = self.output_dir / 'images/right' / filename
                                cv2.imwrite(str(filepath), img_data['image'])
                                all_data['right_images'].append({
                                    'timestamp': timestamp,
                                    'filename': filename,
                                    'frame_id': img_data['frame_id']
                                })
                                counters['right_images'] += 1

                        elif connection.topic == '/sonar3d/pointcloud':
                            pc_data = self.extract_pointcloud(msg, timestamp)
                            if pc_data:
                                filename = f"{counters['pointclouds']:06d}_{timestamp}.pkl"
                                filepath = self.output_dir / 'pointcloud' / filename
                                with open(filepath, 'wb') as f:
                                    pickle.dump(pc_data, f)
                                all_data['pointclouds'].append({
                                    'timestamp': timestamp,
                                    'filename': filename,
                                    'frame_id': pc_data['frame_id']
                                })
                                counters['pointclouds'] += 1

                        elif connection.topic == '/sonar3d/range/ui/compressed':
                            img_data = self.extract_compressed_image(msg, timestamp, 'sonar_range')
                            if img_data:
                                filename = f"{counters['sonar_range']:06d}_{timestamp}.jpg"
                                filepath = self.output_dir / 'sonar/range' / filename
                                cv2.imwrite(str(filepath), img_data['image'])
                                counters['sonar_range'] += 1

                        elif connection.topic == '/sonar3d/intensity/ui/compressed':
                            img_data = self.extract_compressed_image(msg, timestamp, 'sonar_intensity')
                            if img_data:
                                filename = f"{counters['sonar_intensity']:06d}_{timestamp}.jpg"
                                filepath = self.output_dir / 'sonar/intensity' / filename
                                cv2.imwrite(str(filepath), img_data['image'])
                                counters['sonar_intensity'] += 1

                        elif connection.topic == '/auip/vehicle_interface/sensor/imu':
                            imu_data = self.extract_imu(msg, timestamp)
                            all_data['imu'].append(imu_data)
                            counters['imu'] += 1

                        elif connection.topic == '/auip/vehicle_interface/sensor/depth':
                            counters['depth'] += 1

                        elif connection.topic == '/auip/vehicle_interface/odometry':
                            odom_data = self.extract_odometry(msg, timestamp)
                            all_data['odometry'].append(odom_data)
                            counters['odometry'] += 1

                        elif connection.topic == '/auip/camera/forward/left/camera_info':
                            if 'left' not in all_data['camera_info']:
                                cam_info = self.extract_camera_info(msg, timestamp)
                                all_data['camera_info']['left'] = cam_info
                                counters['camera_info_left'] += 1

                        elif connection.topic == '/auip/camera/forward/right/camera_info':
                            if 'right' not in all_data['camera_info']:
                                cam_info = self.extract_camera_info(msg, timestamp)
                                all_data['camera_info']['right'] = cam_info
                                counters['camera_info_right'] += 1

                        total_processed = sum(counters.values())
                        if total_processed % 1000 == 0 and total_processed > 0:
                            print(f"已处理: {total_processed} 条消息...")

                    except KeyboardInterrupt:
                        print("\n\n用户中断!")
                        raise
                    except Exception as e:
                        if sum(counters.values()) < 5:
                            print(f"处理消息时出错: {e}")
                        continue

            self.save_statistics(counters, all_data)

            print("\n" + "=" * 60)
            print("提取完成! 统计信息:")
            print("=" * 60)
            for key, count in counters.items():
                print(f"  {key}: {count}")
            print("=" * 60)

            return all_data, counters

        except Exception as e:
            print(f"\n严重错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def save_statistics(self, counters, all_data):
        """保存统计和元数据"""
        with open(self.output_dir / 'statistics.json', 'w') as f:
            json.dump(counters, f, indent=2)

        timestamp_index = {
            'left_images': all_data['left_images'],
            'right_images': all_data['right_images'],
            'pointclouds': all_data['pointclouds'],
            'imu_count': len(all_data['imu']),
            'odometry_count': len(all_data['odometry'])
        }
        with open(self.output_dir / 'timestamp_index.json', 'w') as f:
            json.dump(timestamp_index, f, indent=2)

        if all_data['camera_info']:
            with open(self.output_dir / 'camera_info/calibration.json', 'w') as f:
                json.dump(all_data['camera_info'], f, indent=2)

        with open(self.output_dir / 'imu/imu_data.json', 'w') as f:
            json.dump(all_data['imu'], f, indent=2)

        with open(self.output_dir / 'odometry/odometry_data.json', 'w') as f:
            json.dump(all_data['odometry'], f, indent=2)

        print(f"\n✓ 元数据已保存到 {self.output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("ROS2 Bag数据提取工具")
    print("=" * 60)

    # 你的bag文件路径（已配置）
    BAG_PATH = r'D:\bishe\2025_08_26_15_28_18_best\2025_08_26_15_28_18_0.db3'
    OUTPUT_DIR = './extracted_data'

    bag_path = Path(BAG_PATH)
    if not bag_path.exists():
        print(f"\n 错误: bag文件不存在!")
        print(f"   路径: {bag_path.absolute()}")
        input("\n按回车键退出...")
        exit(1)

    print(f"✓ bag文件: {bag_path.name}")
    print(f"✓ 路径: {bag_path.parent}")

    try:
        extractor = RosbagExtractor(BAG_PATH, OUTPUT_DIR)
        all_data, counters = extractor.extract_all()

        if all_data is not None:
            print("\n" + "=" * 60)
            print(" 数据提取完成! ")
            print(f" 输出目录: {Path(OUTPUT_DIR).absolute()}")
            print("=" * 60)
            print("\n下一步:")
            print("  运行: python synchronize_data.py")
        else:
            print("\n 提取失败")
    except Exception as e:
        print(f"\n 程序异常: {e}")

    input("\n按回车键退出...")