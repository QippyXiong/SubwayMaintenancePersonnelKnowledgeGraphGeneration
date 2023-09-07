# # public packages
# from pathlib import Path
# from os import path
# import json5
# from threading import Thread
# from torch.utils.data import DataLoader


# # src pckages
# from controller import KGConstructionController, NerTypes, ReTypes
# from controller.contruction_model_types import NerModelComposition, ReModelComposition
# from utils.animator import Animator

# # 配置文件目录
# # 以下路径宏都是相对于项目目录 SubwayMaintenancePersonnelKnowledgeGraphGeneration
# # 基础路径只会在这里设置，以避免问题，Controller类会从基础路径出发按相对路径查找文件

# class Implement(KGConstructionController):
#     r"""
#     实现KGConstructionController，其中需要实现一些定制接口
#     """

#     def __init__(self) -> None:
#         PROJ_DIR    = Path(path.dirname(__file__)).parent # 项目目录，这里是通过 src上一级目录锁定，
#         MODEL_DIR   = PROJ_DIR.joinpath('models') # models文件夹
#         DATASET_DIR = PROJ_DIR.joinpath('data', 'datasets')
#         CONFIG_DIR  = PROJ_DIR.joinpath('config')
#         super().__init__(MODEL_DIR, MODEL_DIR, DATASET_DIR, CONFIG_DIR)

#         self.ani = Animator('step', 'loss')

#     def train_ner_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]):
#         self.ani.add(total_step, loss)
#         if total_step + 1 == self.ani.x_lim[1]:
#             self.ani.close()

#     def before_ner_train(self, composition: NerModelComposition, loader: DataLoader) -> None:
#         num_epochs = composition.model.params.train_params.num_epochs
#         self.ani.clear()
#         self.ani.x_lim=[0, num_epochs * len(loader)]
    
#     def before_re_train(self, composition: ReModelComposition, loader: DataLoader) -> None:
#         num_epochs = composition.model.params.train_params.num_epochs
#         self.ani.clear()
#         self.ani.x_lim=[0, num_epochs * len(loader)]
    
#     def train_re_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]) -> None:
#         self.ani.add(total_step, loss)

# from typing import Union

from fastapi import FastAPI
from typing import Union


# def main():
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

import uvicorn

from pathlib import Path
import os
from database.utils import load_excel_file_to_graph
from database import connect_to_neo4j
import json5

if __name__ == '__main__':
    # uvicorn.run("main:app", port=5000, log_level="info")

    # 连接到neo4j
    NEO4J_FILE_PATH = Path(os.path.dirname(__file__)).parent.joinpath('config', 'database_config', 'neo4jdb.json')
    with open(NEO4J_FILE_PATH, 'r', encoding='UTF-8') as fp:
        neo4j = json5.load(fp)
    connect_to_neo4j(**neo4j)

    # 导入database excel文件
    FILE_PATH = Path(os.path.dirname(__file__))
    FILE_PATH = FILE_PATH.parent.joinpath('data', 'database', 'Synthesis', '维保人员数据.xlsx')
    print(FILE_PATH)
    load_excel_file_to_graph(FILE_PATH)