r"""
此模块中定义 construction_model_controller中所需要的一些类型
原因是先前的文件中类型定义太多，主要类又写得比较长，看得很烦
"""

from typing import Any, Type, TypeVar, Union, Optional
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
    model : ReModelTypes
    embedder: ReEmbedderTypes


@dataclass
class NerModelComposition:
    model: NerModelTypes
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

from datasets import CLNerDataSet, DgreNerDataset, DgreReDataset, DuIEReDataSet, DuIEData, DuIENerDataset
from datasets import CLNerLabelTranser, DgreNerLabelTranser, DuIENerLabelTranser
from datasets import DgreReLabelTranser, DuIESchema, DuIEReLabelTranser


r"""
不同数据集有不同的标签信息，此处根据数据集名字得到标签转换器的类型
"""
dataset_ner_label_transer_mapping: dict[str, type] = {
    'Chinese-Literature-NER-RE-Dataset': CLNerLabelTranser,
    'dgre': DgreNerLabelTranser,
    'DuIE2.0': DuIENerLabelTranser
}


dataset_re_label_transer_mapping: dict[str, type] = {
    'Chinese-Literature-NER-RE-Dataset': None, # @TODO: Finish bolly shit
    'dgre': DgreReLabelTranser, 
    'DuIE2.0': DuIEReLabelTranser
}

# ================================= begin tool functions ===============================================

@dataclass
class NerEntity:
    entity: str
    entity_type: str
    entity_pos: Optional[tuple[int, int]]

@dataclass
class Relation:
    subject: NerEntity
    relation: str
    object: NerEntity

def convert_label_seq_to_entity_pos(sentence: str, label_seq: list[str]) -> list[NerEntity]:
    r""" 将 B-I 格式的标记序列转换成用（实体，实体类型，位置）标记数组 """
    begin_index = 0
    entity = ''
    entity_type = ''
    entity_array: list[NerEntity] = []
    for i, label in enumerate(label_seq):
        if label[0] == 'B':
            begin_index = i
            if entity != '':
                entity_array.append(NerEntity(entity, entity_type, (begin_index, i-1)))
            entity = sentence[i]
            entity_type = label[2:]
        elif label[0] == 'I':
            entity += sentence[i]
            entity_type = label[2:]
        elif label[0] == 'O' and entity != '':
            entity_array.append(NerEntity(entity, entity_type, (begin_index, i-1)))
            entity = ''
        
    if entity != '':
        entity_array.append(NerEntity(entity, entity_type, (begin_index, len(sentence)-1)))
    return entity_array

def convert_entity_pos_to_label_seq(sentence: str, entity_array: list[NerEntity])->list[str]:
    r""" 转换成 B-I 格式的标记顺序 """
    label_seq: list[str] = [ 'O' for a in sentence]
    begin_index, end_index = 0, 0
    for entity in entity_array:
        if entity.entity_pos:
            label_seq[entity.entity_pos[0]] = 'B-' + entity.entity_type
            for j in range(entity.entity_pos[0]+1, entity.entity_pos[1]+1):
                label_seq[j] = 'I-' + entity.entity_type
        else:
            begin_index = sentence.find(entity.entity)
            while begin_index != -1: # 可能有不止一个
                end_index = begin_index + len(entity.entity)
                label_seq[begin_index] = 'B-' + entity.entity_type
                for j in range(begin_index + 1, end_index):
                    label_seq[j] = 'I-' + entity.entity_type
                begin_index = sentence.find(sentence, end_index)
    return label_seq


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
    

class NerDuIEToBertBilstmEncoder:

    def __init__(self, embedder: BertBilstmNerEmbedder) -> None:
        self.embedder = embedder
        self.transer = DuIENerLabelTranser()
    
    def __call__(self, data: DuIEData) -> Any:
        entity_array = []
        for spo in data.spo_list:
            if spo.subject: # 存在为 '' 的情况
                entity_array.append( NerEntity( entity=spo.subject, entity_type=spo.subject_type, entity_pos=None ) )
            if spo.object['@value']: # 存在为 '' 的情况
                entity_array.append( NerEntity( entity=spo.object['@value'], entity_type=spo.object_type['@value'], entity_pos=None ) )
        labels = convert_entity_pos_to_label_seq(data.text, entity_array)
        # padding
        seq_len = self.embedder.seq_len
        labels = labels[0:seq_len - 2]
        label_ids = self.transer.label2id(labels)
        label_ids = [0] + label_ids +  [0] * (seq_len - len(label_ids) - 1)
        return self.embedder(data.text), tensor(label_ids, dtype=long)


class ReDuIEToBertSoftmaxEncoder:
    def __init__(self, embedder: SoftmaxReEmbedder) -> None:
        self.embedder = embedder
        self.transer = DuIEReLabelTranser()

    def __call__(self, sentence: str, spo: DuIEData.SpoData) -> Any:
        return self.embedder(sentence, spo.subject, spo.object['@value']), tensor(self.transer.label2id(spo.predicate), dtype=long)