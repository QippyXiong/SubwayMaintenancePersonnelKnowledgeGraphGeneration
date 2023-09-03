# public packages
from pathlib import Path
from os import path
from typing import Any
import json5
from threading import Thread
import torch
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as ner_report
from torch.utils.data import DataLoader


# src pckages
from controller import KGConstructionController, NerTypes, ReTypes
from controller.contruction_model_types import DataLoader, NerModelComposition, ReModelComposition
from utils.animator import Animator

# 配置文件目录
# 以下路径宏都是相对于项目目录 SubwayMaintenancePersonnelKnowledgeGraphGeneration
# 基础路径只会在这里设置，以避免问题，Controller类会从基础路径出发按相对路径查找文件

class Implement(KGConstructionController):

    def __init__(self) -> None:
        PROJ_DIR    = Path(path.dirname(__file__)).parent # 项目目录，这里是通过 src上一级目录锁定，
        MODEL_DIR   = PROJ_DIR.joinpath('models') # models文件夹
        DATASET_DIR = PROJ_DIR.joinpath('data', 'datasets')
        CONFIG_DIR  = PROJ_DIR.joinpath('config')
        super().__init__(MODEL_DIR, MODEL_DIR, DATASET_DIR, CONFIG_DIR)

        self.ani = Animator('step', 'loss')

    def train_ner_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]):
        self.ani.add(total_step, loss)

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


imple = Implement()
imple.init_re(ReTypes.BERT_SOFTMAX)
task = lambda: imple.train_re(0)
imple.save_re_model(0)
t = Thread(target=task)
t.start()
imple.ani.show()
t.join()
imple.save_re_model(0)


