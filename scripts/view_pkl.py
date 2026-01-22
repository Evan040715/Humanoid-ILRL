import joblib
import numpy as np
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = _REPO_ROOT / "resources/feedforward_data/violin_left15_poses.pkl"

def print_structure(d, indent=0):
    """递归打印字典结构的辅助函数"""
    prefix = " " * indent
    
    if isinstance(d, dict):
        print(f"{prefix}📂 字典包含 Keys: {list(d.keys())}")
        for k, v in d.items():
            print(f"{prefix} - Key '{k}': ", end="")
            if isinstance(v, dict):
                print("⬇️ (嵌套字典)")
                print_structure(v, indent + 4)
            elif isinstance(v, np.ndarray):
                print(f"Shape {v.shape}, Range [{np.min(v):.2f}, {np.max(v):.2f}]")
            elif isinstance(v, list):
                print(f"List (Length {len(v)})")
            else:
                print(f"{type(v)}")
    else:
        print(f"{prefix} {type(d)}")

def inspect_data():
    file_path = Path(FILE_PATH)
    if not file_path.exists():
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"📂 正在加载: {file_path}")
    
    try:
        data = joblib.load(file_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print(f"✅ 读取成功！开始分析结构...")
    print("="*40)
    print_structure(data)
    print("="*40)

if __name__ == "__main__":
    inspect_data()