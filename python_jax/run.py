#!/usr/bin/env python
"""
可执行的运行脚本：显式动力学有限元求解器
"""
import sys
from pathlib import Path

# 确保可以导入当前目录的模块
sys.path.insert(0, str(Path(__file__).parent))

from main import main

if __name__ == "__main__":
    main()
