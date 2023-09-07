r"""
复现论文：

"""

from torch import nn
from transformers import AlbertModel
from pathlib import Path

class SpeacialTaggingJointAlbertModel(nn.Module):

    def __init__(self, albert_root_url: str) -> None: