"""MiaForge 数据层：真实确定性数据集（digits/iris）+ 参考用合成生成器。

作者：晨星
"""

from .dataset import MiaData, load_dataset
from .generator import SyntheticMiaGenerator

__all__ = ["MiaData", "SyntheticMiaGenerator", "load_dataset"]
