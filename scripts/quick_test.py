"""
快速验证测试 - 确保部署环境正确
LLM 辅助: 本文件由 GitHub Copilot 辅助生成
"""

import sys
import subprocess


def test_python_version():
    """测试Python版本"""
    print("=" * 60)
    print("测试 1: Python 版本")
    print("=" * 60)
    
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python版本符合要求 (>= 3.8)")
        return True
    else:
        print("✗ Python版本不符合要求，需要 Python 3.8+")
        return False


def test_package_imports():
    """测试必要的包是否安装"""
    print("\n" + "=" * 60)
    print("测试 2: 必要包导入")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'ultralytics': 'Ultralytics YOLO',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
    }
    
    all_passed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} ({package})")
        except ImportError:
            print(f"✗ {name} ({package}) - 未安装")
            all_passed = False
    
    return all_passed


def test_gpu_availability():
    """测试GPU可用性"""
    print("\n" + "=" * 60)
    print("测试 3: GPU 可用性")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  - GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA 版本: {torch.version.cuda}")
            print(f"  - GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # 测试简单的GPU操作
            x = torch.randn(100, 100).cuda()
            y = x @ x
            print(f"✓ GPU 计算测试通过")
            return True
        else:
            print("⚠ CUDA 不可用，将使用CPU训练（速度较慢）")
            return False
    
    except Exception as e:
        print(f"✗ GPU测试失败: {e}")
        return False


def test_yolo_basic():
    """测试YOLO基本功能"""
    print("\n" + "=" * 60)
    print("测试 4: YOLO 基本功能")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # 创建一个随机图像
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 加载预训练模型（nano版本，快速测试）
        print("加载 YOLOv8n-seg 模型...")
        model = YOLO("yolov8n-seg.pt")
        
        # 测试预测
        print("测试预测功能...")
        results = model.predict(test_image, verbose=False)
        
        print("✓ YOLO 基本功能测试通过")
        print(f"  - 模型加载成功")
        print(f"  - 预测功能正常")
        
        return True
    
    except Exception as e:
        print(f"✗ YOLO测试失败: {e}")
        return False


def test_file_structure():
    """测试项目文件结构"""
    print("\n" + "=" * 60)
    print("测试 5: 项目文件结构")
    print("=" * 60)
    
    import os
    from pathlib import Path
    
    required_files = [
        'requirements.txt',
        'environment.yml',
        'scripts/train.py',
        'scripts/test.py',
        'scripts/evaluate.py',
        'scripts/preprocess_data.py',
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - 文件不存在")
            all_exist = False
    
    return all_exist


def test_directory_structure():
    """测试目录结构"""
    print("\n" + "=" * 60)
    print("测试 6: 目录结构")
    print("=" * 60)
    
    from pathlib import Path
    
    required_dirs = [
        'scripts',
        'test',
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"⚠ {dir_path}/ - 目录不存在（将自动创建）")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 检查数据集目录（可选）
    if Path('dataset').exists():
        print(f"✓ dataset/ - 数据集目录存在")
    else:
        print(f"⚠ dataset/ - 数据集目录不存在（请下载数据集后放入此目录）")
    
    return True


def print_summary(results):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print("\n" + "-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    print("-" * 60)
    
    if passed == total:
        print("\n🎉 所有测试通过! 环境配置正确，可以开始训练。")
        return True
    else:
        print("\n⚠ 部分测试失败，请检查环境配置。")
        print("建议:")
        print("1. 运行: pip install -r requirements.txt")
        print("2. 或使用: conda env create -f environment.yml")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("南极动物语义分割 - 环境验证测试")
    print("=" * 60)
    print()
    
    # 运行所有测试
    results = {
        'Python版本': test_python_version(),
        '包导入': test_package_imports(),
        'GPU可用性': test_gpu_availability(),
        'YOLO功能': test_yolo_basic(),
        '文件结构': test_file_structure(),
        '目录结构': test_directory_structure(),
    }
    
    # 打印总结
    success = print_summary(results)
    
    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
