# shared/setup.py
import os
import sys

def add_project_root():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"root"))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)  # 優先して先頭に入れる
