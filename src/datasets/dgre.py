r"""
    dgre数据集的处理代码
"""

from torch.utils.data import Dataset
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Callable, Any, Union, Iterable
import json

r"""
    {
        "text": "62号汽车故障报告综合情况:故障现象:加速后，丢开油门，发动机熄火。", 
        "labels": ["发动机", "熄火", "部件故障"], 
        "id": "0_0"
    }
"""
@dataclass_json
@dataclass
class DgreData:
    text: str
    labels: list[str] # (subject, object, rel)
    id: str


class DgreReLabelTranser:
    r"""
    文字和id转换器

    通过`id2label`和`label2id`可以完成关系名称和关系id的转换
    """
    def __init__(self) -> None:
        self.labels = [ '没关系', '性能故障', '部件故障', '检测工具', '组成']
        self.label_dict = { name : i for i, name in enumerate(self.labels) }

    def id2label(self, ids: Union[int, list[int]]) -> Union[str, list[str]]:
        if isinstance(ids, Iterable):
            return [ self.labels[i] for i in ids ]
        return self.labels[ids]
    
    def label2id(self, labels: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(labels, str): # 此处不可用Iterable来判断，因为str也是可以遍历的
            return self.label_dict[labels]
        return [ self.label_dict[i] for i in labels]


class DgreReDataset(Dataset):
    r"""
    dgre数据集中关系抽取部分
    """
    def __init__(self, file_path, embedder: Callable[[DgreData], Any] = None) -> None:
        super().__init__()
        self.embedder = embedder
        self.transer = DgreReLabelTranser()
        with open(file_path, mode='r', encoding='UTF-8') as fp:
            self.data = fp.readlines()

    def __getitem__(self, index):
        data:DgreData = DgreData.from_json(self.data[index])
        if self.embedder is None:
            return data
        else:
            return self.embedder(data)
    
    def __len__(self):
        return len(self.data)
    
    def id2label(self, ids: Union[int, list[int]]) -> Union[str, list[str]]:
        return self.transer(ids)
    
    def label2id(self, labels: Union[str, list[str]]) -> Union[int, list[int]]:
        return self.transer(labels)



class DgreNerLabelTranser:
    r"""
    用于将实体识别数据集中的标注作转换

    """
    def __init__(self):
        self.labels = [ 'O', 'B-故障设备', 'I-故障设备', 'B-故障原因', 'I-l故障原因']
        self.label_dict = { name : i for i, name in enumerate(self.labels) }

    def id2label(self, ids: Union[int, list[int]]) -> Union[str, list[str]]:
        if isinstance(ids, Iterable):
            return [ self.labels[i] for i in ids ]
        return self.labels[ids]
    
    def label2id(self, labels: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(labels, str): # 此处不可用Iterable来判断，因为str也是可以遍历的
            return self.label_dict[labels]
        return [ self.label_dict[i] for i in labels]

class DgreNerDataSet(Dataset):
    r"""
    dgre数据集中实体识别的部分
    """
    def __init__(self, file_path, embedder: Callable[[str, list[int]], Any]) -> None:
        super().__init__()
        with open(file_path, mode='r', encoding='UTF-8') as fp:
            self.data = fp.readlines() # 用字符串直接存储反而节省空间
        self.transer = DgreNerLabelTranser()
        self.embedder = embedder

    def __getitem__(self, index) -> Union[tuple[str, list[int]], Any]:
        data_dict = json.loads(self.data[index])
        sentence = ''.join(data_dict['text'])
        labels = self.transer.label2id(data_dict['labels'])
        if self.embedder:
            return self.embedder(sentence, labels)
        return sentence, labels
    
    def id2label(self, ids: Union[int, list[int]]) -> Union[str, list[str]]:
        return self.transer(ids)
    
    def label2id(self, labels: Union[str, list[str]]) -> Union[int, list[int]]:
        return self.transer(labels)
    