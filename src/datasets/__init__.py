r"""
    数据集处理包，每有一个数据集就构建一个新文件继承DataSet特别处理此数据集的内容
"""
from .duie import DuIERelationDataSet, DuIEData, DuIESchema
from .dgre import DgreReDataset, DgreData, DgreReLabelTranser, DgreNerDataset, DgreNerLabelTranser

from .chinese_literature import NerDataSet as CLNerDataSet, NerLabelTranser as CLNerLabelTranser

# @TODO: 所有label transer实现接口 label_transer_interface

from typing import Union
class label_transer_interface:

    def id2label() -> Union[str, list[str]]:
        ...

    def label2id() -> Union[int, list[int]]:
        ...

    def __len__() -> int:
        ...
    
    def get_labels():
        ...