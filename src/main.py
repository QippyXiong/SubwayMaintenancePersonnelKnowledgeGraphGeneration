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
PROJ_DIR    = Path(path.dirname(__file__)).parent # 项目目录，这里是通过 src上一级目录锁定，


class Implement(KGConstructionController):
    r"""
    实现KGConstructionController，其中需要实现一些定制接口
    """

    def __init__(self) -> None:
        MODEL_DIR   = PROJ_DIR.joinpath('models') # models文件夹
        DATASET_DIR = PROJ_DIR.joinpath('data', 'datasets')
        CONFIG_DIR  = PROJ_DIR.joinpath('config')
        super().__init__(MODEL_DIR, MODEL_DIR, DATASET_DIR, CONFIG_DIR)

        self.ani = Animator('step', 'loss')

    def train_ner_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]):
        self.ani.add(total_step, loss)
        if total_step + 1 == self.ani.x_lim[1]:
            self.ani.save_img(PROJ_DIR.joinpath("ner_train_loss.png"))
            # self.ani.close()

    def before_ner_train(self, composition: NerModelComposition, loader: DataLoader) -> None:
        num_epochs = composition.model.params.train_params.num_epochs
        self.ani.clear()
        self.ani.x_lim=[0, num_epochs * len(loader)]

    def before_re_train(self, composition: ReModelComposition, loader: DataLoader) -> None:
        num_epochs = composition.model.params.train_params.num_epochs
        self.ani.clear()
        self.ani.x_lim=[0, num_epochs * len(loader)]
        self.save_step = 50

    def train_re_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]) -> None:
        self.ani.add(total_step, loss)

        if total_step % self.save_step == 0:
            self.save_step = 1000
            self.save_re_model(0)
            self.ani.save_img(PROJ_DIR.joinpath('re_train_loss.png'))

        if total_step + 1 == self.ani.x_lim[1]:
            self.ani.save_img(PROJ_DIR.joinpath('re_train_loss.png'))
            # self.ani.close()

from typing import Union, Dict

# from typing import Union


import uvicorn

from pathlib import Path
import os
from database.utils import load_excel_file_to_graph, EntityQueryByAtt, parse_record_to_dict, \
    RelQueryByEnt, getRelEnt, get_time_key, GetEntAttribute, CreateEnt, DeleteEnt, UpdateEnt, UpdateRel, CreateRel
from database import connect_to_neo4j
import json5
from database import MaintenanceWorker, Capacity, CapacityRate, MaintenanceRecord
from neomodel import db, RelationshipManager, Relationship, StructuredNode, DateTimeFormatProperty, DateTimeProperty

if __name__ == '__main__':

    # 连接到neo4j
    NEO4J_FILE_PATH = Path(os.path.dirname(__file__)).parent.joinpath('config', 'database_config', 'neo4jdb.json')
    with open(NEO4J_FILE_PATH, 'r', encoding='UTF-8') as fp:
        neo4j = json5.load(fp)
    connect_to_neo4j(**neo4j)

    # att = GetEntAttribute("MaintenanceWorker")
    class_name = "MaintenanceWorker"
    attr = {'id': "m0006"}
    attrs = {'level': "高级"}
    ret = MaintenanceWorker.nodes.filter(**attr)
    for e in ret:
        a = Capacity.nodes.get(name='轨道维修')
        CreateRel(a, e, "CapacityRate",attrs)
        # print(edge.all())
        # edge.save()
        # edge = rel.relationship(a)
        # for key, value in attrs.items():
        #     setattr(edge, key, value)
        # # print(parse_record_to_dict(edge))
        # edge.save()
        # UpdateRel(a, e, "CapacityRate", attrs)
    # new_attr = {'birth': '1994-08-27'}
    # {name: "3", rule: "1"}
    # ret = CreateEnt(class_name, attr)
    # print(ret)
    # ret = DeleteEnt(class_name, attr)
    # print(ret)
    # ret = DeleteEnt(class_name, attr)
    # print(ret)
    # ret = UpdateEnt(class_name, attr, new_attr)
    # print(ret)
    # uvicorn.run("server:app", port=5200, log_level="info")

    # impl = Implement()
    # # id = impl.init_ner(NerTypes.BERT_BILSTM_CRF)
    # # t = Thread(target=lambda: impl.train_ner(id, num_workers=1))
    # # t.start()
    # # impl.ani.show()
    # # t.join()
    # # print(impl.valid_ner(id, output_dict=False))
    # # impl.save_ner_model(id)
    # # exit(-1)
    # # id = impl.load_ner_model('fuyandashi.BertBilstmNerModel')
    # # print(impl.valid_ner(id, output_dict=False))
    # # id = impl.load_re_model('test.SoftmaxReModel')
    # # t = Thread(target=lambda: impl.train_re(id, num_workers=1))
    # # t.start()
    # # impl.ani.show()
    # # t.join()
    # # impl.save_re_model(id)
    # # print(impl.valid_re(id, output_dict=False))
    # # impl.save_re_model(id)
    # # entities, rels = impl.ner_re_joint_predicate("在导师阵容方面，英达有望联手《中国喜剧王》选拔新一代笑星", 0, 0)
    # # print(rels)
    # impl.load_re_model('test.SoftmaxReModel')
    # print(impl.valid_re(0))
    # impl.save_re_model(0)