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
from ner_model import BertBilstmNerModel, BertBilstmNerModelParams, BertBilstmNerEmbedder
from datasets import DgreData, DgreReDataset, CLNerDataSet, DgreReLabelTranser
from utils.animator import Animator

from re_model import SoftmaxReModel, SoftmaxReModelParams, SoftmaxReEmbedder
from datasets import DuIERelationDataSet as DuIEDataSet, DuIEData, DuIESchema
from utils.animator import Animator
# 配置文件目录
# 以下路径宏都是相对于项目目录 SubwayMaintenancePersonnelKnowledgeGraphGeneration
# 基础路径只会在这里设置，以避免问题，Controller类会从基础路径出发按相对路径查找文件

PROJ_DIR    = Path(path.dirname(__file__)).parent # 项目目录，这里是通过 src上一级目录锁定，
MODEL_DIR   = PROJ_DIR.joinpath('models') # models文件夹
DATASET_DIR = PROJ_DIR.joinpath('data', 'datasets')
CONFIG_DIR  = PROJ_DIR.joinpath('config')


class DgreEncoder:
    r""" dgre数据集的输出不能直接由embdder实现，所以构建这个类作为embdder """
    def __init__(self, embdder: SoftmaxReEmbedder) -> None:
        self.embdder = embdder
        self.transer = DgreReLabelTranser()
    
    def __call__(self, data: DgreData) -> Any:
        _data = self.embdder(data.text, data.labels[0], data.labels[1])
        label = torch.tensor( self.transer.label2id(data.labels[2]), dtype=torch.long)
        return _data, label


def train_re():
    # 读取默认参数文件
    with open( path.join(CONFIG_DIR, 're_params.json'), 'r', encoding='UTF-8' ) as fp:
        defaut_params : SoftmaxReModelParams = SoftmaxReModelParams.from_dict( json5.load(fp) )
    net = SoftmaxReModel(params=defaut_params, bert_root_dir=str( MODEL_DIR.joinpath('bert') ))
    embedder = SoftmaxReEmbedder( MODEL_DIR.joinpath('bert', defaut_params.hyper_params.bert), defaut_params.hyper_params.seq_len )
    encoder = DgreEncoder(embedder)
    train_loader = DataLoader(
        DgreReDataset( DATASET_DIR.joinpath('dgre', 're_data', 'train.txt'), embedder=encoder),
        batch_size=net.params.train_params.batch_size,
        shuffle=True,
        num_workers=2
    )
    ani = Animator('step', 'loss', x_lim=[0, len(train_loader) * net.params.train_params.num_epochs])
    task = lambda: SoftmaxReModel.train_epochs( 
        net, 
        train_loader, 
        each_step_callback= lambda epoch, total_step, loss, pred, label: ani.add(total_step, loss)
    )
    t = Thread(target=task)
    t.start()
    ani.show()
    t.join()
    # 验证模型
    valid_loader = DataLoader( 
        DgreReDataset( DATASET_DIR.joinpath('dgre', 're_data', 'dev.txt'), embedder=encoder),
        batch_size=net.params.train_params.batch_size
    )
    preds, targets = SoftmaxReModel.validate( net, valid_loader=valid_loader)
    transer = DgreReLabelTranser()
    preds = [ transer.id2label(pred) for pred in preds]
    targets = [ transer.id2label(target) for target in targets ]
    report = classification_report(targets, preds)
    net.set_report(report)
    # 保存模型
    net.params.name = 'dgre_re v0.1'
    net.save( str( MODEL_DIR.joinpath('re') ) )


def tarin_ner():
    with open( CONFIG_DIR.joinpath('ner_params.json'), 'r', encoding='UTF-8') as fp:
        params_dict = json5.load(fp)
    params : BertBilstmNerModelParams = BertBilstmNerModelParams.from_dict(params_dict)
    net = BertBilstmNerModel(
        params=params, 
        bert_root_dir=MODEL_DIR.joinpath('bert')
    )
    embedder = BertBilstmNerEmbedder(
        bert_url=MODEL_DIR.joinpath('bert', params.hyper_params.bert), 
        seq_len=params.hyper_params.seq_len
    )
    train_loader = DataLoader(
        dataset=CLNerDataSet( DATASET_DIR.joinpath('Chinese-Literature-NER-RE-Dataset', 'ner', 'train.txt'), embedder=embedder),
        batch_size=params.train_params.batch_size,
        shuffle=True,
        num_workers=2
    )
    ani = Animator('step', 'loss', x_lim=[0, len(train_loader)*params.train_params.epochs])
    task = lambda: BertBilstmNerModel.train_epochs(
        net, 
        train_loader, 
        each_step_handler=lambda epoch, total_step, loss, pred, label: ani.add(total_step, loss)
    )
    t = Thread(target=task)
    t.start()
    ani.show()
    t.join()
    valid_set = CLNerDataSet( DATASET_DIR.joinpath('Chinese-Literature-NER-RE-Dataset', 'ner', 'validation.txt'), embedder=embedder)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=net.params.train_params.batch_size,
        shuffle=False,
        num_workers=2
    )
    preds, targets = BertBilstmNerModel.valid(net, valid_loader)
    preds = [ valid_set.id2label(pred) for pred in preds ]
    targets = [ valid_set.id2label(target) for target in targets ]
    report_text = ner_report(targets, preds)
    net.set_report(report_text)
    net.save(MODEL_DIR.joinpath('ner'))


if __name__ == '__main__':
    train_re()