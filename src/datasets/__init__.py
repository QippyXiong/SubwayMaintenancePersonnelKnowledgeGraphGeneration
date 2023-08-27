r"""
    数据集处理包，每有一个数据集就构建一个新文件继承DataSet特别处理此数据集的内容
"""
from .duie import DuIERelationDataSet, DuIEData, DuIESchema
from .dgre import DgreReDataset, DgreData, DgreReLabelTranser

from .chinese_literature import NerDataSet as CLNerDataSet