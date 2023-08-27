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


    def __init__(self, schema_file_path, max_relation_num = None) -> None:
        r"""
        Args:
            `schema_file_path`: DuIE数据集下的 duie_schema.json 文件的路径
        """
        with open(schema_file_path, 'r', encoding='UTF-8') as fp:
            self.schemas : list[DuIESchema.DuIESchemaData] = []
            self.relations = set()
            self.relations.add('无关系')
            lines = fp.readlines()
            if max_relation_num is not None: lines = lines[0:max_relation_num - 1]
            for line in lines:
                data: DuIESchema.DuIESchemaData = DuIESchema.DuIESchemaData.from_json(line)
                self.schemas.append(data)
                self.relations.add(data.predicate)
                self.relations2id = { rel : i for i, rel in enumerate(self.relations) }


    def id2rel(self, indices: Union[int, list[int]]) -> Union[str, list[str]]:
        if not isinstance(indices, Iterable):
            return self.relations[indices]
        return [ self.relations[i] for i in indices ]
    
    def rel2id(self, relations: Union[str, list[str]]) -> int:
        if isinstance(relations, str):
            return self.relations2id[relations]
        return [ self.relations2id[rel] for rel in relations ]


class DuIERelationDataSet(Dataset):
    r""" 数据集 DuIE2.0，此数据集类不会将整个文件读入内存 """

    def __init__(self, path: str, dataset_type: str, encoder: Callable[[str, DuIEData.SpoData], Any] = None, schema: DuIESchema = None, max_length = None) -> None:
        r"""
            类型为 train, dev, test2
            encoder: (text, SpoData) -> Any
            若有encoder，getitem 返回 encoder 的返回值，没有encoder直接返回 DuIEData
        """
        super().__init__()
        self.encoder = encoder
        self.dataset_type = dataset_type
        self.path = path
        self.data = []
        with open(path, 'r', encoding='UTF-8') as fp:
            lines = fp.readlines()
            if max_length is not None: lines = lines[0:max_length]
            for line in tqdm(lines, f'reading data'):
                data: DuIEData = DuIEData.from_json(line)
                for spo in data.spo_list:
                    if spo.predicate in schema.relations:
                        self.data.append((data.text, spo))
        return
        if dataset_type not in ('train', 'dev', 'test2'):
            raise TypeError('没有此类型的训练数据集')
        self.dataset_type = dataset_type
        self.encoder = encoder
        self.fp = None
        self.path = path

        if not os.path.exists( str(path) + '.preprocessed'):
            self.fp = open( str(path) + '.preprocessed', 'w', encoding='UTF-8')
            # 用预处理来换下一次运行的速度
            self.fp.write('                        \n') # 第一行保存数据个数
            length = 0
            with open(path, 'r', encoding='UTF-8') as fp:
                lines = fp.readlines()
                for line in tqdm(lines, f'Preprocessing { self.dataset_type } data'):
                    data: DuIEData = DuIEData.from_json(line)
                    for spo in data.spo_list:
                        self.fp.write( json.dumps({ 'text' : data.text, 'spo': spo.to_json() }) + '\n' )
                        length += 1
            self.fp.seek(0)
            self.fp.write(str(length))
            self.fp.close()
        
        self.fp = open( str(path) + '.preprocessed', 'r', encoding='UTF-8')
        self.length = int(self.fp.readline().strip())
        self.fp.close()
    
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
    
        if self.fp.closed:
            self.fp = open( str(self.path) + '.preprocessed', 'r', encoding='UTF-8')
        r""" 采用文件指针读入而不是存储到运行内存中 """
        for i, line in enumerate(self.fp):
            if i == index + 1: # 第一行是数据集大小
                self.fp.seek(0)
                line_data = json.loads(line)
                if self.encoder is not None:
                    return self.encoder( line_data['text'], DuIEData.SpoData.from_json(line_data['spo']) )
                else:
                    return line_data['text'], DuIEData.SpoData.from_dict(line_data['spo'])
    
    def close(self):
        if hasattr(self, 'fp'): self.fp.close()
        
