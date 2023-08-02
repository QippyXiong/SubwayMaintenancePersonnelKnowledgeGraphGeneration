from torch import nn, Tensor
import torch
from torchcrf import CRF
from transformers import BertConfig, BertForMaskedLM, logging as trans_loger
import os

current_path = os.path.dirname(__file__)
# SubwayMaintenancePersonnelKnowledgeGraphGeneration Dir
project_path = ''.join([ item + os.path.sep for item in current_path.split(os.path.sep)[:-3]]) # ../../..

model_dir = os.path.join(project_path, 'ChineseLiteratureKG', 'model')

# chinese-wwm-ext dir
CN_BERT_DIR = os.path.join(model_dir, 'chinese-bert-wwm-ext')


# 可以改写，反正我也没咋写
class NerModel(nn.Module):
    r""" 用于命名体识别的模型，采用Bert + BiLSTM + CRF """

    def __init__(self, num_labels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        trans_loger.set_verbosity_error() # 用于取消在控制台的参数未完全使用的警告打印
        self.bert = BertForMaskedLM.from_pretrained(CN_BERT_DIR)
        # bert输出句子特征的Tensor大小
        self.bert_hidden_size = self.bert.config.hidden_size # class : BertConfig
        self.bert.config.max_length
        trans_loger.set_verbosity_warning() # 重新启用警告打印

        self.lstm_hidden_size = 128
        # BiLSTM
        self.bilstm = nn.LSTM(self.bert_hidden_size, self.lstm_hidden_size, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
        # Linear
        self.linear = nn.Linear(self.lstm_hidden_size*2, num_labels)
        # CRF
        self.crf = CRF(num_labels, True)
        
    
    def forward(input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor):
        """ 计算结果 BERT-BiLSTM-CRF """
        

    default_save_path = os.path.join(model_dir, 'ner_model.bin')

    def save(self, save_path : str = default_save_path):
        torch.save(self.state_dict(), save_path)

    
    def load(self, save_path : str = default_save_path):
        self.load_state_dict(torch.load( save_path ))


