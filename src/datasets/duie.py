from torch.utils.data import Dataset
from typing import Callable, Any, Union, Iterable
from json5 import loads
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import os
from tqdm import tqdm

r"""
data is like:
{
    "text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下", 
    "spo_list": [
        {
            "predicate": "作者", 
            "object_type": {
                "@value": "人物"
            }, 
            "subject_type": "图书作品", 
            "object": {
                "@value": "冰火未央"
            }, 
            "subject": "邪少兵王"
        }
    ]
}
"""


@dataclass_json
@dataclass
class DuIEData:

    @dataclass_json
    @dataclass
    class SpoData:
        predicate: str
        object_type: dict # key '@value' is the name of object_type
        subject_type: str
        object: dict # key '@value' is the value of objet
        subject: str

    text: str
    spo_list: list[SpoData]


class DuIESchema:

    # {"object_type": {"@value": "语言"}, "predicate": "官方语言", "subject_type": "国家"}
    @dataclass_json
    @dataclass
    class DuIESchemaData:
        object_type: dict
        predicate: str
        subject_type: str

    def __init__(self) -> None:
        self.triplets =  [('人物', '毕业院校', '学校'), ('电视综艺', '嘉宾', '人物'), ('娱乐人物', '配音', '人物'), ('影视作品', '主题曲', '歌曲'), ('企业/品牌', '代言人', '人物'), ('歌曲', '所属专辑', '音乐专辑'), (' 人物', '父亲', '人物'), ('图书作品', '作者', '人物'), ('影视作品', '上映时间', 'Date'), ('人物', '母亲', '人物'), ('学科专业', '专业代码', 'Text'), ('机构', '占地面积', 'Number'), ('行政区', '邮政编码', 'Text'), ('影视作品', '票房', 'Number'), ('企业', '注册资本', 'Number'), ('文学作品', '主角', '人物'), ('人物', '妻子', '人物'), ('影视作品', '编剧', '人物'), ('行政区', '气候', '气候'), ('歌曲', '歌手', '人物'), ('娱乐人物', '获奖', '奖项'), ('学校', '校长', '人物'), ('企业', '创始人', '人物'), ('国家', '首都', '城市'), ('人物', '丈夫', '人物'), ('历史人物', '朝代', 'Text'), ('娱乐人物', '饰演', '人物'), (' 行政区', '面积', 'Number'), ('企业', '总部地点', '地点'), ('人物', '祖籍', '地点'), ('行政区', '人口数量', 'Number'), ('影视作品', '制片人', '人物'), ('学科专业', '修业年限', 'Number'), ('景点', '所在城市', '城市'), ('企业', '董事长', '人物'), ('歌曲', '作词', '人物'), ('影视作品', '改编自', '作品'), ('影视作品', '出品公司', '企业'), ('影视作品', '导演', '人物'), ('歌曲', '作曲', '人物'), ('影视作品', '主演', '人物'), ('电视综艺', '主持人', '人物'), ('机构', '成立日期', 'Date'), ('机构', '简称', 'Text'), ('地点', '海拔', 'Number'), ('历史人物', '号', 'Text'), ('人物', '国籍', '国家'), ('国家', '官方语言', ' 语言')]
    

class DuIEReLabelTranser:

    def __init__(self, max_length: int = None) -> None:
        self.labels = ['毕业院校', '嘉宾', '配音', '主题曲', '代言人', '所属专辑', '父亲', '作者', '上映时间', '母亲', '专业代码', '占地面积', '邮政编码', '票房', '注册资本', '主角', '妻子', '编剧', '气候', '歌手', '获奖', '校长', '创始人', '首都', '丈夫', '朝代', '饰演', '面积', '总部地点', '祖籍', '人口数量', '制片人', '修业年限', '所在城市', '董事长', '作词', '改编自', '出品公司', '导演', '作曲', '主演', '主持人', '成立日期', '简称', '海拔', '号', '国籍', '官方语言']
        if max_length: self.labels = self.labels[0: max_length]
        self.labels_dict = { label: i for i, label in enumerate(self.labels) }
    
    def id2label(self, indices: Union[int, list[int]]) -> Union[str, list[str]]:
        if not isinstance(indices, Iterable):
            return self.labels[indices]
        return [ self.labels[i] for i in indices ]
    
    def label2id(self, relations: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(relations, str):
            return self.labels_dict[relations]
        return [ self.labels_dict[rel] for rel in relations ]

    def __len__(self) -> int:
        return len(self.labels)
    
    def get_labels(self):
        return self.labels


class DuIERelationDataSet(Dataset):
    r""" 数据集 DuIE2.0，此数据集类不会将整个文件读入内存 """

    def __init__(self, file_path: str, encoder: Callable[[str, DuIEData.SpoData], Any] = None, max_length: int = None) -> None:
        r"""
        Args:
            `encoder`: (text, SpoData) -> Any
            `max_length`: 最多有几种label，因为这个数据集比较可观（49种label），输入后会舍弃掉后面的label
        """
        super().__init__()
        self.encoder = encoder
        self.dataset_type = encoder
        self.data = []
        self.transer = DuIEReLabelTranser(max_length)
        with open(file_path, 'r', encoding='UTF-8') as fp:
            lines = fp.readlines()
            for line in tqdm(lines, f'reading data'):
                data: DuIEData = DuIEData.from_json(line)
                for spo in data.spo_list:
                    if spo.predicate in self.transer.get_labels():
                        self.data.append((data.text, spo))
        return
    
    def __len__(self) -> int:
        return len(self.data)
        return self.length

        r"""
            数据集大小：
                train:  171135
                dev:    20652
                test2:  101239
        """
        len_dict = {
            'train':  171135,
            'dev':    20652,
            'test2':  101239
        }
        return len_dict[self.dataset_type]
    
    def __getitem__(self, index) -> Union[tuple[str, DuIEData.SpoData], Any]:
        return self.data[index] if self.encoder is None else self.encoder(*self.data[index])

    def close(self):
        if hasattr(self, 'fp'): self.fp.close()
        if self.data: del self.data # call gc
        
