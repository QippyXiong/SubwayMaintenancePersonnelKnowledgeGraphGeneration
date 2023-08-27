r"""
    此模型是使用 Bert+BiLSTM+CRF 的命名实体识别模型
"""

from torch import nn, Tensor, tensor, long
import torch
from torchcrf import CRF
from transformers import BertModel, logging as trans_loger, get_linear_schedule_with_warmup, AutoTokenizer
import os
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from torch.optim import AdamW
from typing import Callable, Union, Sequence, Optional
from tqdm import tqdm

@dataclass_json
@dataclass
class NerModelParams:
    r""" 模型参数集合 """
    @dataclass_json
    @dataclass
    class TrainParams:
        batch_size          : int  
        epochs              : int  
        warm_up_proportion  : float
        bert_lr             : float
        bert_decay          : float
        crf_lr              : float
        crf_decay           : float
        other_lr            : float
        other_decay         : float
        eps                 : float
        drop_out            : float

    @dataclass_json
    @dataclass
    class HyperParams:
        bert               : str
        num_labels         : int
        seq_len            : int
        lstm_hidden_size   : int
        lstm_hidden_layers : int

    hyper_params: HyperParams
    train_params: TrainParams
    name: str
    description: str = ''
    dataset: str = '' # 模型训练用数据集的名字


class NerEmbedder(object):
    r""" 用于将单个句子处理成输入数据 """
    def __init__(self, bert_url: str, seq_len: Optional[int] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(bert_url)
        self.seq_len = seq_len
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.pad_token = pad_token

    def __call__(self, sentence: str, batch: bool = False, seq_len: Optional[int] = None) -> dict:
        r""" 非 data_loader 加载的时候（单个句子判断），设置 batch 为 True """
        if seq_len is None: seq_len = self.seq_len
        # perData = self.tokenizer(sentence, return_offsets_mapping=True, max_length=self.seq_len, truncation=True)
        perData = self.tokenizer(sentence, padding='max_length', max_length=seq_len, truncation=True, return_offsets_mapping=True)
        # self.data.append({ key: Tensor(perData[key]) for key in perData })
        result = {}
        for key in perData:
            if batch:
                result[key] = tensor([perData[key]], dtype=long)
            else:
                result[key] = tensor(perData[key], dtype=long)
    
        return result


class NerModel(nn.Module):
    r""" 
    用于命名体识别的模型，采用Bert + BiLSTM + CRF，通过save，load函数来存储加载模型
    此模型暂时还不支持Albert，后续会添加支持
    """

    def __init__(self, params: NerModelParams, bert_root_dir, *args, **kwargs) -> None:
        r"""
        Args:
            params: Model Parameters
            bert_root_dir: bert model parent directory, load bert will use `bert_root_dir/bert` for `bert` in `params.hyper_params`
        """
        super().__init__(*args, **kwargs)

        trans_loger.set_verbosity_error() # 用于取消在控制台的参数未完全使用的警告打印

        self.params = params
        hps = params.hyper_params
        # if 'bert' in parms_config.keys():
        bert_url = os.path.join(bert_root_dir, hps.bert)
        self.bert = BertModel.from_pretrained(bert_url)
        
        # bert输出句子特征的Tensor大小
        self.bert_hidden_size = self.bert.config.hidden_size # class : BertConfig

        trans_loger.set_verbosity_warning() # 重新启用警告打印

        # BiLSTM
        self.bilstm = nn.LSTM(
            self.bert_hidden_size,
            hps.lstm_hidden_size,
            hps.lstm_hidden_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.params.train_params.drop_out
        )
        # Linear
        self.linear = nn.Linear(hps.lstm_hidden_size*2, hps.num_labels)
        # CRF
        self.crf = CRF(hps.num_labels, True)
        
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, labels: Tensor = None) -> tuple[Tensor, Tensor]:
        r""" 
            计算结果 BERT-BiLSTM-CRF，data需包含`input_ids`，`attention_mask`，`token_type_ids`
            labels是对应的结果序列
            返回值为`(preds, loss)`
        """
        # 还有position_ids未输入
        # bert_output: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        seq_out : Tensor = self.bert(input_ids=input_ids, attention_mask=attention_mask.bool(), token_type_ids=token_type_ids)[0]
        seq_out, _ = self.bilstm(seq_out)
        seq_out = self.linear(seq_out)
        logits = self.crf.decode(seq_out, mask=attention_mask.bool())

        loss = None
        if labels is not None:
            loss = -self.crf(seq_out, tags=labels, mask=attention_mask.bool(), reduction='mean')
        return logits, loss
    
    
    def set_report(self, report_text: str) -> None:
        r""" 设置要保存的验证集验证信息 """
        self.report = report_text

    def get_report(self) -> Union[str, None]:
        r""" return validating report text if has else return `None` """
        if hasattr(self, 'report'):
            return self.report
        else: return None

    def save(self, save_dir: str)->None:
        r"""
            会存储模型文件model.bin和模型config文件config.json，实际写入到一个文件夹中
            save_dir: 存储文件夹目录，默认为model文件夹下
            会根据名字属性 name 创建文件夹，
        """
        model_path = os.path.join(save_dir, self.params.name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.state_dict(), os.path.join(model_path, "ner_model.bin"))
        with open(os.path.join(model_path, "params.json"), 'w', encoding='UTF-8') as fp:
            params_json = self.params.to_json()
            fp.write(params_json)
        if self.get_report() is not None:
            with open(os.path.join(model_path, "report.txt"), 'w', encoding='UTF-8') as fp:
                fp.write(self.get_report())


    @staticmethod
    def load(model_dir : str, bert_root_dir: str):
        r""" 加载模型，model_dir为模型文件夹路径 """
        if not os.path.exists(os.path.join(model_dir, "params.json")):
            raise FileNotFoundError(f'[{__package__}:Error]missing params.json file, check if this is the real model directory')
        with open(os.path.join(model_dir, "params.json"), 'r', encoding='UTF-8') as fp:
            params = json.load(fp)
        params = NerModelParams.from_dict(params)
        ner = NerModel(params, bert_root_dir)
        ner.load_state_dict(torch.load( os.path.join(model_dir, "ner_model.bin") ))
        return ner
    

    @staticmethod
    def train_epochs(
        net, 
        train_loader: Sequence, 
        device = 'cuda:0',
        each_step_handler : Callable[[int, int, float, list[list[int]], list[list[int]] ], None] = None
    ) -> None:
        r""" 
        train according to params.train_params

        Args:
            train_loader: should implement `__len__` method, item return `( {'input_ids，'attention_mask'，'token_type_ids'}, label )`, all is `Tensor`
            device: cpu or gpu
            each_step_handler: `(epoch, total_step, loss, pred, label)->None`, take care of batch size
        """

        # deal parameters
        bert_params = [ (name, param) for name, param in net.named_parameters() if 'bert' in name ]
        crf_params  = [ (name, param) for name, param in net.named_parameters() if 'crf' in name ]
        other_params= [ (name, param) for name, param in net.named_parameters() if 'bert' not in name and 'crf' not in name ]

        # print([name for name, param in bert_params])
        # print([name for name, param in crf_params])
        # print([name for name, param in other_params])
        no_decay = ( 'bias', 'layerNorm' ) # name of no decay params

        tps: NerModelParams.TrainParams = net.params.train_params
        grouped_params = [
            { # bert with decay
                'params': [ param for name, param in bert_params if not any(nd in name for nd in no_decay) ],
                'weight_decay': tps.bert_decay,
                'lr': tps.bert_lr
            },
            { # bert without decay
                'params': [ param for name, param in bert_params if any(nd in name for nd in no_decay) ],
                'weight_decay': 0.,
                'lr': tps.bert_lr
            },

            { # other with decay
                'params': [ param for name, param in crf_params if not any(nd in name for nd in no_decay) ],
                'weight_decay': tps.crf_decay,
                'lr': tps.crf_lr
            },
            { # other without decay
                'params': [ param for name, param in crf_params if any(nd in name for nd in no_decay) ],
                'weight_decay': 0.,
                'lr': tps.crf_lr
            },

            { # other with decay
                'params': [ param for name, param in other_params if not any(nd in name for nd in no_decay) ],
                'weight_decay': tps.other_decay,
                'lr': tps.other_lr
            },
            { # other without decay
                'params': [ param for name, param in other_params if any(nd in name for nd in no_decay) ],
                'weight_decay': 0.,
                'lr': tps.other_lr
            }
        ]

        optimizer = AdamW(params=grouped_params, lr=tps.other_lr, eps=tps.eps)
        total_steps = tps.epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * tps.warm_up_proportion), total_steps)

        net.to(device)
        for epoch in range(1, tps.epochs + 1):
            net.train()
            iteration = tqdm(train_loader)
            for input, label in iteration:
                for key in input.keys():
                    input[key] = input[key].to(device)
                label = label.to(device)

                pred, loss = net(
                    input_ids=input['input_ids'],
                    attention_mask=input['attention_mask'],
                    token_type_ids=input['token_type_ids'],
                    labels=label
                )
                # input['labels'] = label
                # pred, loss = net(**input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iteration.set_description('train(loss %3.3f)'% loss.item())

                # (epoch, step, loss, pred: list (batch_size, ), label: list (batch_size, ))
                each_step_handler(epoch, (epoch-1)*len(iteration) + iteration.n, loss.item(), pred, label.tolist())

    
    @staticmethod
    def valid(
        net,
        valid_loader: Sequence,
        device = 'cuda:0'
    ) -> tuple[list[list[int]], list[list[int]]]:
        r"""
        Args:
        
        Returns: (preds, targets)
            preds: preds of the label index for valid loader input sentence
            targets: target
        """

        net.to(device)
        net.eval()
        iteration = tqdm(valid_loader)
        preds, targets = [], []
        with torch.no_grad():
            for input, label in iteration:
                for key in input.keys():
                    input[key] = input[key].to(device)
                label = label.to(device)

                pred, loss = net(
                    input_ids=input['input_ids'],
                    attention_mask=input['attention_mask'],
                    token_type_ids=input['token_type_ids'],
                    labels=label
                )
                iteration.set_description('validation(loss: %3.3f)'%loss.item())

                preds += pred
                label_list = label.tolist()
                for i in range(len(label_list)):
                    label_list[i] = label_list[i][:len(pred[i])] # turncate to the same length
                targets += label_list

        return preds, targets
