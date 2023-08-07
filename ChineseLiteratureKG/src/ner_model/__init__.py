from torch import nn, Tensor
import torch
from torchcrf import CRF
from transformers import BertModel, logging as trans_loger
import os
import json

current_path = os.path.dirname(__file__)
# SubwayMaintenancePersonnelKnowledgeGraphGeneration Dir
project_path = ''.join([ item + os.path.sep for item in current_path.split(os.path.sep)[:-3]]) # ../../..

MODEL_DIR = os.path.join(project_path, 'ChineseLiteratureKG', 'model')

# chinese-wwm-ext dir
CN_BERT_DIR = os.path.join(MODEL_DIR, 'chinese-bert-wwm-ext')


# 可以改写，反正我也没咋写
class NerModel(nn.Module):
    r""" 用于命名体识别的模型，采用Bert + BiLSTM + CRF，通过save，load函数来存储加载模型 """

    def __init__(self, config: dict, name : str = 'ner', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.name = name # 存储模型时文件夹名字即为此字符串
        self.dummpy_param = nn.Parameter(torch.empty(0), requires_grad=False) # 方便用于定位device
        self.valid_report = None # validation报告
        self.description = None # 关于模型的一段描述

        parms_config : dict = config['model_parms']
        self.seq_len            = parms_config['seq_len']
        self.lstm_hidden_size   = parms_config['lstm_hidden_size']
        self.num_labels         = parms_config['num_labels']
        self.lstm_hidden_layers = 1
        if 'lstm_hidden_layers' in parms_config.keys():
            self.lstm_hidden_layers = parms_config['lstm_hidden_layers']

        trans_loger.set_verbosity_error() # 用于取消在控制台的参数未完全使用的警告打印

        if 'bert' in parms_config.keys():
            self.bert = BertModel.from_pretrained(os.path.join(MODEL_DIR, parms_config['bert']))
        else:
            self.bert = BertModel.from_pretrained(CN_BERT_DIR)
        
        # bert输出句子特征的Tensor大小
        self.bert_hidden_size = self.bert.config.hidden_size # class : BertConfig
        if self.bert.config.max_length < self.seq_len:
            print("[Warning] model input sequence length bigger than bert max input length in NerModel")
        
        trans_loger.set_verbosity_warning() # 重新启用警告打印
        # BiLSTM
        self.bilstm = nn.LSTM(
            self.bert_hidden_size,
            self.lstm_hidden_size,
            self.lstm_hidden_layers,
            bidirectional=True,
            batch_first=True #
        )
        # Linear
        self.linear = nn.Linear(self.lstm_hidden_size*2, self.num_labels)
        # CRF
        self.crf = CRF(parms_config['num_labels'], True)
        
    
    def forward(self, data: dict, labels: Tensor = None) -> tuple[Tensor, Tensor]:
        r""" 
            计算结果 BERT-BiLSTM-CRF，data需包含input_ids，attention_mask，token_type_ids
            labels是对应的结果序列
            返回值为(logits, loss)
        """
        # 还有position_ids未输入
        # bert_output: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        
        seq_out : Tensor = self.bert(input_ids=data['input_ids'], attention_mask=data['attention_mask'].bool(), token_type_ids=data['token_type_ids'])[0]
        # 从bert输出seq_out.shape = (batch_size, seq_len, bert_size)

        # state = self.create_hidden_state(batch_size=seq_out.shape[0])
        # seq_out = seq_out.permute(1, 0, 2) # (seq_len, batch_size, bert_size)
        # lstm_out = torch.zeros((seq_out.shape[0], seq_out.shape[1], 2*self.lstm_hidden_size), device=self.dummpy_param.device) # (seq_len, batch_size, bert_size)
        # for i in range(0, self.seq_len):
        #     word_label, state = self.bilstm(seq_out[i], state) # (batch_size, lstm_hidden_size)
        #     lstm_out[i] = word_label
        # lstm_out = lstm_out.permute(1, 0, 2) # (batch_size, seq_len, lstm_hidden_size)

        seq_out, _ = self.bilstm(seq_out)
        seq_out = self.linear(seq_out)
        logits = self.crf.decode(seq_out, mask=data['attention_mask'].bool())
        loss = None
        if labels is not None:
            loss = -self.crf(seq_out, tags=labels, mask=data['attention_mask'].bool(), reduction='mean')
        return logits, loss

    default_save_dir = os.path.join(MODEL_DIR, 'ner')

    def save(self, save_dir: str = default_save_dir)->None:
        r"""
            会存储模型文件model.bin和模型config文件config.json，实际写入到一个文件夹中
            save_dir: 存储文件夹目录，默认为model文件夹下
            会根据名字属性 name 创建文件夹，
        """
        model_path = os.path.join(save_dir, self.name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.state_dict(), os.path.join(model_path, "ner_model.bin"))
        with open(os.path.join(model_path, "config.json"), 'w', encoding='UTF-8') as fp:
            json.dump(self.config, fp)
        if self.valid_report is not None:
            with open(os.path.join(model_path, "report.txt"), 'w', encoding='UTF-8') as fp:
                fp.write( str(self.valid_report) )
        if self.description is not None:
            with open(os.path.join(model_path, "description.txt"), 'w', encoding='UTF-8') as fp:
                fp.write( str(self.valid_report) )

    def load(self, save_dir: str = default_save_dir):
        r"""
            根据目录和名字加载模型

            这个只能加载 config['model_params'] 相同的，建议使用 load_ner_model 函数加载模型
        """
        model_path = os.path.join(save_dir, self.name)
        self.load_state_dict(torch.load( os.path.join(model_path, "ner_model.bin") ))
        with open(os.path.join(model_path, "config.json"), 'r', encoding='UTF-8') as fp:
            self.config = json.load(fp)


def load_ner_model(name: str, dir : str = NerModel.default_save_dir) -> NerModel:
    r""" 加载模型 """
    model_path = os.path.join(dir, name)
    config = {}
    with open(os.path.join(model_path, "config.json"), 'r', encoding='UTF-8') as fp:
        config = json.load(fp)
    ner = NerModel(config, name=name)
    ner.load_state_dict(torch.load( os.path.join(model_path, "ner_model.bin") ))
    return ner
