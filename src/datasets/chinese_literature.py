from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torch import tensor, long
import os
from typing import Optional, Callable, Any, Union
import json


class NerLabelTranser():
    r""" 将实体类型文本和定义数值之间进行转换，在NerDataSet中定义了一个静态成员，不需要导入 """
    
    def __init__(self) -> None:
        r"""
            初始化，遍历过程中会构建字典tag_label_dict

            此处写死了dict
        """
        self.tag_label_dict = { # 此数据集格式命名不规范
            "O": 0,
            "B-Time": 1,
            "I-Time": 2,
            "B-Person": 3,
            "I-Person": 4,
            "B-Thing": 5,
            "I-Thing": 6,
            "B-Location": 7,
            "I-Location": 8,
            "B-Metric": 9,
            "I-Metric": 10,
            "B-Organization": 11,
            "I-Organization": 12,
            "B-Abstract": 13,
            "I-Abstract": 14,
            "B-Physical": 15,
            "I-Physical": 16,
            "B-Term": 17,
            "I-Term": 18
        }
        # 此处将 `_` 改成 `-` 是为了避免格式问题的 report 报错
        self.tag_labels = [
            "O",
            "B-Time",
            "I-Time",
            "B-Person",
            "I-Person",
            "B-Thing",
            "I-Thing",
            "B-Location",
            "I-Location",
            "B-Metric",
            "I-Metric",
            "B-Organization",
            "I-Organization",
            "B-Abstract",
            "I-Abstract",
            "B-Physical",
            "I-Physical",
            "B-Term",
            "I-Term"
        ]
    
    def id2label(self, indices: list[int]) -> list[str]:
        return [ self.tag_labels[i] for i in indices]
    
    def label2id(self, labels: list[str]) -> list[int]:
        return [ self.tag_label_dict[i] for i in labels]

    def label2id_add(self, labels: list[str]) -> list[int]:
        """ 此方法会将不认识的label添加到字典里，请谨慎使用 """
        ret = []
        for label in labels:
            try:
                ret.append(self.tag_label_dict[label])
            except KeyError: # need to add this label
                temp = str(label)[2:] # 消去B-, I-
                self.tag_labels.append('B-' + temp)
                self.tag_label_dict[self.tag_labels[-1]] = len(self.tag_labels) - 1
                self.tag_labels.append('I-' + temp)
                self.tag_label_dict[self.tag_labels[-1]] = len(self.tag_labels) - 1
                ret.append(self.tag_label_dict[label])
        # if len(ret) != len(labels): assert('fuck you man')
        return ret

    def num_labels(self):
        return len(self.tag_labels)

    def load(self, config_file_path: str) -> None:
        with open(config_file_path, 'r', encoding='UTF-8') as fp:
            obj = json.load(fp)
            self.tag_label_dict = obj['tag_label_dict']
            self.tag_labels = obj['tag_labels']

    def save(self, config_file_path: str) -> None:
        with open(config_file_path, 'w', encoding='UTF-8') as fp:
            json.dump({
                'tag_label_dict': self.tag_label_dict,
                'tag_labels': self.tag_labels
            }, fp)
    

class SentenceDistributionStatics(dict):
    r""" 用来统计长度信息，最大长度，平均长度，长度分布情况。这个类可以帮助我们选择合适的句子长度 """
    def __init__(self) -> None:
        super(SentenceDistributionStatics, self).__init__()
        # self.distrib = {} # { len : count }

    def count(self, len: int):
        try:
            self[len] += 1
        except KeyError:
            self[len] = 1
    
    def average(self)->float:
        sum = 0
        count = 0
        for key in self.keys():
            size = self[key]
            sum += key * size
            count += size
        return sum / count

    def max(self):
        return max(self.keys())

class NerDataSet(Dataset):
    r""" 命名实体识别数据集，数据集构建时不对齐句子长度 """

    def __init__(self, file_path : str, embedder: Callable[[str], Any] = None) -> None:
        r""" 
        Args:
            file_path: 加载文件路径
            embedder: `__getitem__` 方法会调用embedder，若有embedder返回值即为embdder的值
        """
        self.transer = NerLabelTranser() # 转换器，存储了标签字典

        self.data = []
        self.label = []
        self.embedder = embedder
        self.distrib = SentenceDistributionStatics()
        sentence = '' # 单个句子
        tagseq = [] # 单个句子对应的标签序列
        # encoder = BertEncoder(tokenizer, seq_len)
        with open(file_path, mode='r', encoding='UTF-8') as fd:
            for line in fd:
                r""" 数据为每个字和对应的标签在一行，一个句子的结束会空一行 """
                if (len(line) > 3):
                    [char, tag] = line.split(' ')
                    sentence += char
                    tag = tag.replace('_', '-') # 此数据集标号有问题
                    tagseq.append(tag[:-1])  # 除去末尾的回车
                else:
                    # fast tokenizer can turncate itself
                    tag_id_seq = []

                    tag_id_seq = self.transer.label2id(tagseq)

                    if len(tag_id_seq) > embedder.seq_len - 2: # 对齐label
                        tag_id_seq = tag_id_seq[:embedder.seq_len - 2]
                    tag_id_seq = [0] + tag_id_seq # 起始是cls token
                    tag_id_seq += [0] * (embedder.seq_len - len(tag_id_seq)) # padding
                    self.data.append(sentence)
                    self.label.append(tensor(tag_id_seq, dtype=long))

                    # 统计句子长度分布
                    self.distrib.count(len(sentence))

                    sentence = ''
                    tagseq = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[Union[str, Any], list[int]]:
        r"""
        有embedder时返回embedder处理后的数据，没有就直接返回数据

        Returns:
            `embedder(sentence), label` or `sentence, label`
        """
        if self.embedder is not None:
            return self.embedder(self.data[index]), self.label[index]
        else:
            return self.data[index], self.label[index]

    def id2label(self, indices: list[int]) -> list[int]:
        return self.transer.id2label(indices)
        
    def label2id(self, labels: list[str]) -> list[int]:
        return self.transer.label2id(labels)
    
    def num_label(self) -> int:
        return self.transer.num_label()


class ReDataSet(Dataset):
    r""" 这文件格式真不太好写 """
    def __init__(self, 
            encoder: Callable[[str, str, str], dict], 
            sentence_file_path: str,  
            annotation_file_path: str
        ) -> None:
        r""" encoder: (sentence, subject, object)-> { 'input_ids': Tensor, 'attention_mask': Tensor, 'token_type_ids': Tensor } """
        super().__init__()
        pass
    
    def __getitem__(self, index) -> None:
        pass