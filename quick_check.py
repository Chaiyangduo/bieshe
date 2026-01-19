#!/usr/bin/env python3
"""
快速测试脚本 - 用于验证工具是否正常工作
"""

import sys
from pathlib import Path


def check_dependencies():
    """检查依赖是否安装"""
    print("=" * 60)
    print("检查依赖包...")
    print("=" * 60)

    required_packages = {
        'rosbags': 'rosbags',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'scipy': 'scipy'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing.append(package)

    if missing:
        print(f"\n缺少以下包: {', '.join(missing)}")
        print("请运行: bash setup_environment.sh")
        return False
    else:
        print("\n✓ 所有依赖已安装!")
        return True


def verify_bag_path():
    """验证bag路径配置"""
    print("\n" + "=" * 60)
    print("验证配置...")
    print("=" * 60)

    # 读取extract_rosbag_data.py查找BAG_PATH
    script_path = Path(__file__).parent / 'extract_rosbag_data.py'

    with open(script_path, 'r') as f:
        content = f.read()

    if "BAG_PATH = '/path/to/your/rosbag'" in content:
        print("⚠ 警告: BAG_PATH 还未配置!")
        print("请编辑 extract_rosbag_data.py 文件")
        print("修改: BAG_PATH = '/path/to/your/rosbag'")
        print("改为你的实际bag文件路径")
        return False
    else:
        print("✓ BAG_PATH 已配置")
        return True


def show_usage():
    """显示使用说明"""
    print("\n" + "=" * 60)
    print("使用流程:")
    print("=" * 60)
    print("""
步骤1: 配置环境
    bash setup_environment.sh

步骤2: 配置bag路径
    编辑 extract_rosbag_data.py
    修改 BAG_PATH = '/path/to/your/rosbag'

步骤3: 提取数据
    python3 extract_rosbag_data.py

步骤4: 同步数据
    python3 synchronize_data.py

步骤5: 开始标注
    查看 CVAT_Annotation_Guide.md

步骤6: 处理标注
    python3 process_cvat_annotations.py \\
        --annotation_dir cvat_annotations/ \\
        --image_dir extracted_data/annotation_subset/ \\
        --output_dir processed_dataset/

详细说明请查看 README.md
    """)


if __name__ == '__main__':
    print("\n🚀 ROS2 Bag数据处理工具包 - 快速检查\n")

    deps_ok = check_dependencies()
    path_ok = verify_bag_path()

    print("\n" + "=" * 60)
    if deps_ok and path_ok:
        print("✓ 准备就绪! 可以开始处理数据")
        print("  运行: python3 extract_rosbag_data.py")
    else:
        print("⚠ 请先完成上述配置")
    print("=" * 60)

    show_usage()