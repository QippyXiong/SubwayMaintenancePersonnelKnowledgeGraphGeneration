from torch.utils.data import Dataset
from torch import tensor, long
from .BertEmbedder import BertEmbedder
import os
from typing import Optional
import json

current_path = os.path.dirname(__file__)
project_path = ''.join([ item + os.path.sep for item in current_path.split(os.path.sep)[:-3]]) # ../../..

CN_BERT_DIR = os.path.join(project_path, 'ChineseLiteratureKG', 'model', 'chinese-bert-wwm-ext')
NER_DATASET_DIR = os.path.join(project_path, 'data', 'dataset', 'Chinese-Literature-NER-RE-Dataset', 'ner')

CONFIG_DIR = os.path.join(project_path, 'ChineseLiteratureKG', 'config')
# cn_bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(CN_BERT_DIR)

class NerLabelTranser():
    r""" 将实体类型文本和定义数值之间进行转换，在NerDataSet中定义了一个静态成员，不需要导入 """
    CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, 'label_transer.json')
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        r"""
            初始化，遍历过程中会构建字典tag_label_dict

            可以使用保存的配置文件初始化，给config_path赋值NerLabelTranser.CONFIG_FILE_PATH会加载
            config/label_transer.json的内容
        """
        self.tag_labels = ['O']
        self.tag_label_dict = { 'O' : 0 }
        if config_path:
            self.load(config_path)
    
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
                temp = str(label)[2:] # 消去B_, I_
                self.tag_labels.append('B_' + temp)
                self.tag_label_dict[self.tag_labels[-1]] = len(self.tag_labels) - 1
                self.tag_labels.append('I_' + temp)
                self.tag_label_dict[self.tag_labels[-1]] = len(self.tag_labels) - 1
                ret.append(self.tag_label_dict[label])
        # if len(ret) != len(labels): assert('fuck you man')
        return ret

    def num_label(self):
        return len(self.tag_labels)

    def load(self, config_file_path: str = CONFIG_FILE_PATH) -> None:
        with open(config_file_path, 'r', encoding='UTF-8') as fp:
            obj = json.load(fp)
            self.tag_label_dict = obj['tag_label_dict']
            self.tag_labels = obj['tag_labels']

    def save(self, config_file_path: str = CONFIG_FILE_PATH) -> None:
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

    def __init__(self, file_path : str, embedder: BertEmbedder, transer: Optional[NerLabelTranser] = None) -> None:
        r""" 
            由于数据集不是很大，文件内容直接全部加载进内存
            @TODO: 实现lazy loading
        """
        if transer is None:
            self.transer = NerLabelTranser() # 转换器，存储了标签字典
        else:
            self.transer = transer

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
                    tagseq.append(tag[:-1])  # 除去末尾的回车
                else:
                    # fast tokenizer can turncate itself
                    tag_id_seq = []

                    if transer is None: # 初始化此数据集的transer
                        tag_id_seq = self.transer.label2id_add(tagseq)
                    else: # 已有transer就不用初始化了，以避免数据集有问题标签结果添加了
                        tag_id_seq = self.transer.label2id(tagseq)
                    if len(tag_id_seq) > embedder.seq_len - 2:
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

    def __getitem__(self, index):
        return self.embedder(self.data[index]), self.label[index]

    def id2label(self, indices: list[int]) -> list[int]:
        self.transer.id2label(indices)
        
    def label2id(self, labels: list[str]) -> list[int]:
        self.transer.label2id(labels)
    
    def num_label(self) -> int:
        return self.transer.num_label()
    

