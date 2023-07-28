from torch import nn
from transformers import BertConfig, BertForMaskedLM, logging as trans_loger
import os

current_path = os.path.dirname(__file__)
# SubwayMaintenancePersonnelKnowledgeGraphGeneration Dir
project_path = ''.join([ item + os.path.sep for item in current_path.split(os.path.sep)[:-3]]) # ../..

# chinese-wwm-ext dir
CN_BERT_DIR = os.path.join(project_path, 'ChineseLiteratureKG', 'model', 'chinese-bert-wwm-ext')


# 可以改写，反正我也没咋写
class NerModel(nn.Module):
    r""" 用于命名体识别的模型，采用Bert + BiLSTM + CRF """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        trans_loger.set_verbosity_error() # 用于取消在控制台的参数未完全使用的警告打印
        self.bert = BertForMaskedLM.from_pretrained(CN_BERT_DIR)
        # bert输出句子特征的Tensor大小
        self.hidden_size = self.bert.config.hidden_size # class : BertConfig
        trans_loger.set_verbosity_warning() # 重新启用警告打印
        # BiLSTM
        # CRF
    
    def forward(x):
        """ 计算结果 BERT-BiLSTM-CRF """
        pass


