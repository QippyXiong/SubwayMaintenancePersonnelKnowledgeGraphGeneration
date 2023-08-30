r"""
    此为控制保存模型、训练模型、加载模型的类
"""

from pathlib import Path
from enum import Enum
import json5
import sys

# ner models
from ner_model import BertBilstmNerModel, BertBilstmNerModelParams, BertBilstmNerEmbedder

# re models
from re_model import SoftmaxReModel, SoftmaxReModelParams, SoftmaxReEmbedder

class KGConstructionController:
    r"""
    业务逻辑类型，输入只有文件路径，会按照每个文件夹应有的结构来执行业务逻辑

    应有结构长啥样后面再归纳，CONFIG_DIR 可能后续会去掉
    """
    def __init__(
            self,
            MODEL_DIR   : Path,
            DATASET_DIR : Path,
            CONFIG_DIR  : Path,
            ) -> None:
        r"""
        Args:
            `MODEL_DIR`: 模型文件夹所在路径
            `DATASET_DIR`: 数据集文件夹所在路径
            `CONFIG_DIR`: 默认模型参数/配置文件夹所在路径
        """
        self.MODEL_DIR   = MODEL_DIR  
        self.DATASET_DIR = DATASET_DIR
        self.CONFIG_DIR  = CONFIG_DIR 
    
    
    class NerTypes(Enum):
        r""" 枚举类，包含实现了NER模型的种类，用来作为实现多个NER模型的接口 """
        BERT_BILSTM_CRF: 1

    def set_ner(self, net, type, embedder = None)->None:
        r"""
        设置ner模型，这是考虑了后续实现所以按照接口的形式实现，后续大概率此函数会复杂化，
        如果实现了不止一个ner模型的话
        """
        self.ner = net
        self.ner_type = type
        self.ner_embedder = embedder

    def init_ner(self, type: NerTypes)->None:

        NER_TYPES = KGConstructionController.NerTypes
        MODEL_DIR = self.MODEL_DIR
        CONFIG_DIR= self.CONFIG_DIR
        
        if type == NER_TYPES.BERT_BILSTM_CRF:
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
            self.set_ner(net, type=type, embedder=embedder)
        
        else:
            raise Exception('[MODEL TYPE NOT FOUND]')

    class ReTypes(Enum):
        BERT_SOFTMAX: int

    def set_re(self, net, type, embedder = None)->None:
        self.re = net
        self.re_type = type
        self.re_embedder = embedder

    def init_re(self, net, type):

        NER_TYPES = KGConstructionController.NerTypes
        MODEL_DIR = self.MODEL_DIR
        CONFIG_DIR= self.CONFIG_DIR
        
        with open( CONFIG_DIR.joinpath('re_params.json'), 'r', encoding='UTF-8' ) as fp:
            defaut_params : SoftmaxReModelParams = SoftmaxReModelParams.from_dict( json5.load(fp) )
        net = SoftmaxReModel(params=defaut_params, bert_root_dir=str( MODEL_DIR.joinpath('bert') ))
        embedder = SoftmaxReEmbedder( MODEL_DIR.joinpath('bert', defaut_params.hyper_params.bert), defaut_params.hyper_params.seq_len )
