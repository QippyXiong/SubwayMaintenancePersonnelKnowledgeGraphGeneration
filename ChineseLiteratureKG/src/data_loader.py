from torch.utils.data import Dataset
from torch import Tensor
from transformers import BertTokenizer
import os

current_path = os.path.dirname(__file__)
project_path = ''.join([ item + os.path.sep for item in current_path.split(os.path.sep)[:-2]]) # ../..

CN_BERT_DIR = os.path.join(project_path, 'ChineseLiteratureKG', 'model', 'chinese-bert-wwm-ext')
DATASET_DIR = os.path.join(project_path, 'data', 'dataset', 'Chinese-Literature-NER-RE-Dataset', 'ner')

cn_bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(CN_BERT_DIR)

class NerLabelTranser():
    r""" 将实体类型文本和定义数值之间进行转换，在NerDataSet中定义了一个静态成员，不需要导入 """
    
    def __init__(self) -> None:
        r""" 初始化，使用label2id_add方法来在遍历数据的过程中构建字典 """
        self.tag_labels = ['O']
        self.tag_label_dict = { 'O' : 0 }
    
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
        

class NerDataSet(Dataset):
    r""" 命名实体识别数据集，每个句子的长度是不对齐的 """

    transer = NerLabelTranser() # 转换器，存储了标签字典
    id2label = lambda x: NerDataSet.transer.id2label(x)
    label2id = lambda x: NerDataSet.transer.label2id(x)

    def __init__(self, file_path : str) -> None:
        r""" 由于数据集不是很大，直接全部加载进内存 """
        self.data = []
        self.label = []
        sentence = '' # 单个句子
        tagseq = [] # 单个句子对应的标签序列
        first = True
        with open(file_path, mode='r', encoding='UTF-8') as fd:
            for line in fd:
                r""" 数据为每个字和对应的标签在一行，一个句子的结束会空一行 """
                if (len(line) > 3):
                    [char, tag] = line.split(' ')
                    sentence += char
                    tagseq.append(tag[:-1])  # 除去末尾的回车
                else:
                    perData = cn_bert_tokenizer(sentence)
                    self.data.append({ key: Tensor(perData[key]) for key in perData })
                    self.label.append(Tensor(NerDataSet.transer.label2id_add(tagseq)))
                    sentence = ''
                    tagseq = []

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    

