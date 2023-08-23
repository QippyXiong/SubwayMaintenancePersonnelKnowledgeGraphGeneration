from torch import nn, tensor, long
from transformers import BertModel, AutoTokenizer
from typing import Iterable, Union
from dataclasses import dataclass
import torch


@dataclass
class ReModelConfig:
    r"""
        定义模型config

        超参数(hyper params)：
            1. 选用的 Bert 模型路径     bert_url : str
            2. 输入句子长度             seq_len
            3. SoftMax 网络隐藏层的大小 num_hidden_size , 如果多层，使num_hidden_size = [...] 为一个数组，每层的输出大小即为
            4. SoftMax 判断的类别数     num_labels
            5. SoftMax DropOut         dropout

        
        训练参数(training params)：
            1. batch_size
            2. num_epochs
            3. bert_lr
            4. linear_lr
            5. eps
            6. bert_decay
            7. linear_decay
    """
    @dataclass
    class HyperParams:
        # def __init__(self,
        bert_url : str  
        seq_len : int        
        num_hidden_size : int 
        num_labels  : float  
        drop_out : Union[list[float], float]
    
    @dataclass
    class TrainParams:  
        batch_size   : int
        num_epochs   : int
        bert_lr      : float
        linear_lr    : float
        eps          : float
        bert_decay   : float
        linear_decay : float

    hyper_params: HyperParams
    train_params: TrainParams



class ReModel(nn.Module):
    r"""
        普普通通关系抽取，用的 Bert + SoftMax
        [cls]输入句子[sep]主体[sep]客体[sep] 经过Bert处理后得到特征向量，然后作 SoftMax 回归
        思想：告诉人一个句子，句子中包含的主体名字，句子中包含的客体名字，人能判断这两个之间是什么关系

        超参数：
            1. 选用的 Bert 模型 bert
            2. 输入句子长度 seq_len
            3. SoftMax回归网络层数 num_hidden_layers
            4. SoftMax回归网络隐藏层大小 num_hidden_size
        
        训练参数：
            1. batch size
            2. num_epochs
            3. bert_lr
            4. linear_lr
            5. eps
            6. bert_decay
            7. linear_decay
    """
    def __init__(self, config: ReModelConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bert : BertModel = BertModel.from_pretrained(config.hyper_params.bert_url)
        self.tokenizer = AutoTokenizer.from_pretrained(config.hyper_params.bert_url)
        self.sequential = nn.Sequential()
        self.config = config

        hps = self.config.hyper_params

        hidden_sizes = [ self.bert.config.hidden_size ] # input size is hidden_size

        # [input size, hidden1, hidden2, ..., output size]
        if isinstance(hps.num_hidden_size, Iterable):
            for size in hps.num_hidden_size:
                hidden_sizes.append(size)
        else: hidden_sizes.append(hps.num_hidden_size)

        hidden_sizes.append(hps.num_labels) # output size is num_labels

        for i in range( len(hidden_sizes) - 2): # hidden layers
            self.sequential.append( nn.Linear( hidden_sizes[i], hidden_sizes[i+1] ) )
            self.sequential.append( nn.Tanh() ) # avoid derivatives boom
            self.sequential.append( nn.Dropout( p = hps.drop_out[i] if isinstance[hps.drop_out] else hps.drop_out ) )

        self.sequential.append( nn.Linear( hidden_sizes[-2], hidden_sizes[-1] ) ) # output layer
        self.sequential.append( nn.Softmax() )


    def forward(self, input_ids, attention_mask, token_type_ids):
        r""" 输入和bert输入相同输出即为 softmax层输出 """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.sequential(bert_output[1]) # last hidden status
    

    def tokenize(self, sentence: str, subject_entity: str, object_entity: str, predict = False) -> dict:
        r""" 组合entity和句子，返回 { 'input_ids', 'attention_mask', 'token_type_ids' } """
        composed_sentence = f'{subject_entity}[SEP]{object_entity}[SEP]{sentence}' # avoid entity truncation
        data : dict = self.tokenizer(composed_sentence, padding='max_length', max_length=self.config.hyper_params.seq_len, truncation=True)
        for p in data.keys():
            if predict:
                data[p] = tensor([data[p]], dtype=long) # avoid bert batch_size error
            else:
                data[p] = tensor(data[p], dtype=long)
        
        data['attention_mask'] = data['attention_mask'].bool() # attention_mask is byte type
        return data
    

    def predict(self, sentence: str, subject_entity: str, object_entity: str) -> int:
        r""" return the index of label """
        embeddings = self.tokenize(sentence, subject_entity, object_entity, predict=True)
        return torch.argmax( self(embeddings) )
