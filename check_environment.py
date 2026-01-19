#!/usr/bin/env python3
"""
环境诊断脚本 - 检查你当前使用的Python环境
"""

import sys
import os

print("=" * 70)
print("🔍 Python环境诊断")
print("=" * 70)

# 1. Python可执行文件路径
print(f"\n1. Python可执行文件:")
print(f"   {sys.executable}")

# 2. Python版本
print(f"\n2. Python版本:")
print(f"   {sys.version}")

# 3. 是否在虚拟环境中
print(f"\n3. 虚拟环境检测:")
in_venv = hasattr(sys, 'real_prefix') or (
    hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
)
if in_venv:
    print(f"   ✓ 在虚拟环境中")
    print(f"   Base: {sys.base_prefix}")
    print(f"   Venv: {sys.prefix}")
else:
    print(f"   ✗ 不在虚拟环境中 (可能在base或系统Python)")
    print(f"   Prefix: {sys.prefix}")

# 4. site-packages路径
print(f"\n4. 包安装位置 (site-packages):")
import site
for path in site.getsitepackages():
    print(f"   - {path}")

# 5. 检查关键包是否安装
print(f"\n5. 依赖包检测:")
packages = {
    'rosbags': 'rosbags',
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'scipy': 'scipy'
}

for module, package in packages.items():
    try:
        mod = __import__(module)
        location = mod.__file__ if hasattr(mod, '__file__') else 'builtin'
        print(f"   ✓ {package:20s} 已安装")
        print(f"     位置: {location}")
    except ImportError:
        print(f"   ✗ {package:20s} 未安装")

# 6. 环境变量
print(f"\n6. 关键环境变量:")
print(f"   VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '未设置')}")
print(f"   CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', '未设置')}")

print("\n" + "=" * 70)
print("💡 诊断建议:")
print("=" * 70)

if 'venv' in sys.executable.lower():
    print("✓ 你正在使用venv虚拟环境")
    print("  建议: 在这个环境中安装所有依赖")
    print("  命令: pip install rosbags opencv-python numpy scipy")
elif 'conda' in sys.executable.lower() or 'anaconda' in sys.executable.lower():
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"⚠ 你正在使用Anaconda环境: {env_name}")
    if env_name == 'base':
        print("  建议: 创建专门的conda环境，不要用base")
        print("  命令: conda create -n rosbag_env python=3.9")
        print("        conda activate rosbag_env")
    else:
        print("  建议: 在这个conda环境中安装依赖")
        print("  命令: pip install rosbags opencv-python numpy scipy")
else:
    print("⚠ 你正在使用系统Python")
    print("  建议: 使用虚拟环境")
    print("  命令: python -m venv venv")

print("\n如果依赖显示✗未安装，请在当前环境中运行:")
print("  pip install rosbags opencv-python numpy scipy")
print("\n" + "=" * 70)

# 7. 提供快速修复命令
print("\n🔧 快速修复命令:")
print("=" * 70)

if not in_venv and 'anaconda' in sys.executable.lower():
    print("\n你在Anaconda base环境中，建议:")
    print("1. 创建虚拟环境:")
    print("   python -m venv venv")
    print("\n2. 激活虚拟环境:")
    print("   .\\venv\\Scripts\\Activate.ps1  (Windows PowerShell)")
    print("   或")
    print("   venv\\Scripts\\activate.bat  (Windows CMD)")
    print("\n3. 安装依赖:")
    print("   pip install rosbags opencv-python numpy scipy")
elif in_venv:
    print("\n你已经在虚拟环境中! 很好!")
    print("如果包未安装，运行:")
    print("   pip install rosbags opencv-python numpy scipy")
else:
    print("\n建议创建虚拟环境:")
    print("   python -m venv venv")
    print("   .\\venv\\Scripts\\Activate.ps1")
    print("   pip install rosbags opencv-python numpy scipy")

print("=" * 70)
