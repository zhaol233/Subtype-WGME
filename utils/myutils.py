# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 15:50
# @Author  : zhaoliang
# @Description: 深度学习功能模块

import platform
import random
import subprocess


import numpy as np
import torch


cancer_dict = {"CLLE": 6, "ESAD": 2, 'MALY': 3, "OV": 3, "PACA": 6, "PAEN": 6, "RECA": 4, "BRCA":4}


def print_environment_info():
    """
    Prints infos about the environment and the system.
    This should help when people make issues containg the printout.
    """

    print("Environment information:")

    # Print OS information
    print(f"System: {platform.system()} {platform.release()}")

    try:
        print(
            f"Current Version: {subprocess.check_output(['poetry', 'version'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Not using the poetry package")

    # Print commit hash if possible
    try:
        print(
            f"Current Commit Hash: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No git or repo found")


def provide_determinism(seed=0):
    """
    设置随机数种子
    Args:
        seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

