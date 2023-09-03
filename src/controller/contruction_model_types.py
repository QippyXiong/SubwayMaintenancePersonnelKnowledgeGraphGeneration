r"""
此模块中定义 construction_model_controller中所需要的一些类型
原因是先前的文件中类型定义太多，主要类又写得比较长，看得很烦
"""

from typing import Any, Type, TypeVar, Union, Callable
from enum import Enum
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import os
from torch import tensor, long

# ner models
from ner_model import BertBilstmNerModel, BertBilstmNerModelParams, BertBilstmNerEmbedder


# re models
from re_model import SoftmaxReModel, SoftmaxReModelParams, SoftmaxReEmbedder

# =================================== model part ======================================================

ReModelTypes = Union[ SoftmaxReModel, Any ]
NerEmbedderTypes = Union[ BertBilstmNerEmbedder, Any ]
NerModelParamTypes = Union[BertBilstmNerModelParams, Any]

NerModelTypes = Union[ BertBilstmNerModel, Any ]
ReEmbedderTypes = Union[ SoftmaxReEmbedder, Any ]
ReModelParamTypes = Union[SoftmaxReModelParams, Any]


class NerTypes(Enum):
    r""" 枚举类，包含实现了NER模型的种类，用来作为实现多个NER模型的接口 """
    BERT_BILSTM_CRF = BertBilstmNerModel


class ReTypes(Enum):
    BERT_SOFTMAX = SoftmaxReModel


@dataclass
class ReModelComposition:
    model : NerModelTypes
    embedder: ReEmbedderTypes


@dataclass
class NerModelComposition:
    model: ReModelTypes
    embedder: NerEmbedderTypes


# @default_params_files
params_default_file_mapping = {
    NerTypes.BERT_BILSTM_CRF : ('bert_bilstm_crf.json', BertBilstmNerModelParams ),
    ReTypes.BERT_SOFTMAX: ('bert_softmax.json', BertBilstmNerModelParams)
}

def get_default_params_file_name(type: NerTypes):
    return params_default_file_mapping[type]

# ================================ end model part ===================================================


# ================================ datasets part ====================================================

from datasets import CLNerDataSet, DgreNerDataset, DgreReDataset, DuIERelationDataSet
from datasets import CLNerLabelTranser, DgreNerLabelTranser
from datasets import DgreReLabelTranser, DuIESchema


r"""
不同数据集有不同的标签信息，此处根据数据集名字得到标签转换器的类型
"""
dataset_ner_label_transer_mapping: dict[str, type] = {
    'Chinese-Literature-NER-RE-Dataset': CLNerLabelTranser,
    'dgre': DgreNerLabelTranser, # @TODO: Finish dgre ner label transer
    'DuIE2.0': DuIESchema, # @TODO: Finish DuIE ner label transer
}


dataset_re_label_transer_mapping: dict[str, type] = {
    'Chinese-Literature-NER-RE-Dataset': None, # @TODO: Finish bolly shit
    'dgre': DgreReLabelTranser, 
    'DuIE2.0': DuIESchema # @TODO
}

# ================================= begin dataset encoders =============================================
# all this XxxxxEncoder classes below are for data processing before train / valid

class NerCLToBertBilstmEncoder:

    def __init__(self, embedder: BertBilstmNerEmbedder, label_transer: CLNerLabelTranser) -> None:
        self.embedder = embedder
        self.label_transer = label_transer

    def __call__(self, sentence: str, labels: list[str]) -> Any:
        seq_len = self.embedder.seq_len
        if seq_len: # padding labels
            labels = labels[0:seq_len-2]
            labels = self.label_transer.label2id(labels)
            labels = [0] + labels + [0] * (seq_len - len(labels) - 1)
        return self.embedder(sentence), tensor(labels, dtype=long) # embedder will automatically padding sentence
            

class NerDgreToBertBilstmEncoder:

    def __init__(self, embedder: BertBilstmNerEmbedder, label_transer: DgreNerLabelTranser) -> None:
        self.embedder = embedder
        self.label_transer = label_transer

    def __call__(self, sentence, labels):
        seq_len = self.embedder.seq_len
        # padding
        labels = labels[0:seq_len-2]
        label_ids = self.label_transer.label2id(labels)
        label_ids = [0] + label_ids +  [0] * (seq_len - len(label_ids) - 1)
        model_input = self.embedder(sentence)
        labels = tensor(label_ids, dtype=long)
        return model_input, labels


from datasets import DgreData


class ReDgreToSoftmaxEncoder:

    def __init__(self, embedder: SoftmaxReEmbedder, label_transer: DgreReLabelTranser) -> None:
        self.embedder = embedder
        self.label_transer = label_transer
    
    def __call__(self, data: DgreData) -> Any:
        DgreReDataset
        s, o, r = data.labels # (subject, object, relation)
        return self.embedder(data.text, s, o), tensor(self.label_transer.label2id(r), dtype=long)