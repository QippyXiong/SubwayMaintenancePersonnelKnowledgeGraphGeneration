# public packages
from pathlib import Path
from os import path
import json5
from threading import Thread
from torch.utils.data import DataLoader


# src pckages
from controller import KGConstructionController, NerTypes, ReTypes
from controller.contruction_model_types import NerModelComposition, ReModelComposition
from utils.animator import Animator

# 配置文件目录
# 以下路径宏都是相对于项目目录 SubwayMaintenancePersonnelKnowledgeGraphGeneration
# 基础路径只会在这里设置，以避免问题，Controller类会从基础路径出发按相对路径查找文件

class Implement(KGConstructionController):
    r"""
    实现KGConstructionController，其中需要实现一些定制接口
    """

    def __init__(self) -> None:
        PROJ_DIR    = Path(path.dirname(__file__)).parent # 项目目录，这里是通过 src上一级目录锁定，
        MODEL_DIR   = PROJ_DIR.joinpath('models') # models文件夹
        DATASET_DIR = PROJ_DIR.joinpath('data', 'datasets')
        CONFIG_DIR  = PROJ_DIR.joinpath('config')
        super().__init__(MODEL_DIR, MODEL_DIR, DATASET_DIR, CONFIG_DIR)

        self.ani = Animator('step', 'loss')

    def train_ner_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]):
        self.ani.add(total_step, loss)
        if total_step + 1 == self.ani.x_lim[1]:
            self.ani.close()

    def before_ner_train(self, composition: NerModelComposition, loader: DataLoader) -> None:
        num_epochs = composition.model.params.train_params.num_epochs
        self.ani.clear()
        self.ani.x_lim=[0, num_epochs * len(loader)]
    
    def before_re_train(self, composition: ReModelComposition, loader: DataLoader) -> None:
        num_epochs = composition.model.params.train_params.num_epochs
        self.ani.clear()
        self.ani.x_lim=[0, num_epochs * len(loader)]
    
    def train_re_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]) -> None:
        self.ani.add(total_step, loss)

if __name__ == '__main__':

    im = Implement()

    id = im.init_re(ReTypes.BERT_SOFTMAX)
    # im.train_ner(id, num_workers=2)
    print( im.valid_re(id, num_workers=0))