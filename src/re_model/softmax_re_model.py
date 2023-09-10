import os
from dataclasses import dataclass
from os import path
from typing import Iterable, Union, Callable, Optional

import torch
from dataclasses_json import dataclass_json
from torch import nn, tensor, int32, save as torchsave, argmax, load as torchload
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer, BertConfig, logging as trans_loger
from transformers import get_linear_schedule_with_warmup


@dataclass_json
@dataclass
class SoftmaxReModelParams:
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
    
    其他数据：
        1. 模型名称
        2. 训练模型用的数据集的名字
    """
    @dataclass_json
    @dataclass
    class HyperParams:
        bert        : str
        seq_len     : int
        hidden_size : Union[list[int], int]
        num_labels  : int
    
    @dataclass_json
    @dataclass
    class TrainParams:  
        batch_size          : int
        num_epochs          : int
        bert_lr             : float
        linear_lr           : float
        eps                 : float
        bert_decay          : float
        linear_decay        : float
        warm_up_proportion  : float
        drop_out            : Union[list[float], float]

    hyper_params: HyperParams
    train_params: TrainParams
    name: str
    dataset: str


class SoftmaxReEmbedder:
    r""" 负责将文字输入（实体名称和句子）转换为模型的输入 """
    def __init__(self, bert_url, seq_len) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(bert_url)
        self.seq_len = seq_len
    
    def __call__(self, sentence: str, subject_entity: str, object_entity: str, predict = False) -> dict:
        r""" 
        组合entity和句子，返回字典 { 'input_ids', 'attention_mask', 'token_type_ids' }

        Args:
            setence: text sentence，文本句子
            subject_entity: relation head entity，文本头实体
            object_entity: relation tail entity，文本尾实体
        
        Returns:
            `{ str: Tensor}`: `{ 'input_ids', 'attention_mask', 'token_type_ids' }`
        """
        composed_entity = f'{subject_entity}[SEP]{object_entity}' # avoid entity truncation
        data : dict = self.tokenizer.encode_plus(composed_entity, sentence, padding='max_length', max_length=self.seq_len, return_token_type_ids=True, truncation=True)
        for p in data.keys():
            if predict:
                data[p] = tensor([data[p]], dtype=int32) # avoid bert batch_size error
            else:
                data[p] = tensor(data[p], dtype=int32)
        
        data['attention_mask'] = data['attention_mask'].bool() # attention_mask is byte type
        return data

class SoftmaxReModel(nn.Module):
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
    def __init__(self, params: SoftmaxReModelParams, bert_root_dir: str, *args, **kwargs) -> None:
        r"""
        Args:

        """
        super().__init__(*args, **kwargs)

        bert_url = path.join(bert_root_dir, params.hyper_params.bert)

        trans_loger.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_url)
        self.bert_config = BertConfig.from_pretrained(bert_url)

        trans_loger.set_verbosity_warning()

        self.sequential = nn.Sequential()
        self.params = params

        hps = self.params.hyper_params

        hidden_sizes = [ self.bert_config.hidden_size ] # input size is hidden_size

        # [input size, hidden1, hidden2, ..., output size]
        if isinstance(hps.hidden_size, Iterable):
            for size in hps.hidden_size:
                hidden_sizes.append(size)
        else: hidden_sizes.append(hps.hidden_size)

        hidden_sizes.append(hps.num_labels) # output size is num_labels

        drop_out = self.params.train_params.drop_out
        for i in range( len(hidden_sizes) - 2): # hidden layers
            self.sequential.append( nn.Linear( hidden_sizes[i], hidden_sizes[i+1] ) )
            self.sequential.append( nn.ReLU() ) # avoid derivatives boom
            self.sequential.append( nn.Dropout( p = drop_out[i] if isinstance(drop_out, Iterable) else drop_out ) )
        self.sequential.append( nn.Linear( hidden_sizes[-2], hidden_sizes[-1] ) ) # output layer
        # self.sequential.append( nn.Softmax(1) ), CrossEntropyLoss会自动计算Softmax，不需要我计算
        # 由于Softmax的保序性，Softmax前后argmax的值都是一样的，因此CrossEntropyLoss直接包含了Softmax的操作


    def forward(self, input_ids, attention_mask, token_type_ids):
        r""" 输入和bert输入相同输出即为 softmax层输出 """
        # print('input ids', input_ids)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(bert_output[1])
        seq_out = self.sequential(bert_output[1])
        return seq_out # last hidden status
    

    def save(self, model_dir: str) -> None:
        r""" 在 model_dir 文件夹下保存模型 """
        save_dir = path.join(model_dir)

        if not path.exists(save_dir): os.makedirs(save_dir)

        save_list = [
            # filename       # file content
            ( 'params.json' , self.params.to_json() ),
            ( 'report.txt'  , self.get_report() )
        ]

        for name, value in save_list:
            with open( path.join(save_dir, name), "w", encoding="UTF-8" ) as fp:
                if value: fp.write( value )
        
        torchsave( self.state_dict(), path.join(save_dir, 'bert_softmax_re_model.bin') )

    @staticmethod
    def load(model_dir: str, bert_root_dir: str):
        r""" 返回加载的模型 """
        save_dir = path.join(model_dir)
        with open( path.join(save_dir, 'params.json'), 'r', encoding='UTF-8' ) as fp:
            params = SoftmaxReModelParams.from_json( fp.read().strip() )
        
        model = SoftmaxReModel(params, bert_root_dir)
        with open( path.join(save_dir, 'report.txt'), 'r', encoding='UTF-8' ) as fp:
            model.set_report( fp.read() )

        model.load_state_dict( torchload( path.join(save_dir, 'bert_softmax_re_model.bin') ) )

        return model

    
    def set_report(self, report_text: str)->None:
        r""" 保存验证集验证报告结果，验证结果会在保存模型时保存到 report.txt 中 """
        self.report = report_text

    
    def get_report(self)-> Union[str, None]:
        if hasattr(self, 'report'):
            if self.report is not None:
                return self.report
        return None
    
    @staticmethod
    def train_epochs(
        net, 
        train_loader: DataLoader, 
        device='cuda:0', 
        each_step_callback: Optional[Callable[[int, int, float, list[int], list[int]], None]] = None
    )->None:
        r""" 
        Dataset getitem should return ({input_ids, attention_mask, token_type_ids}, label: Tensor)
        
        Args:
            net (SoftmaxReModel): training network, use the net.params.train_params to train
            train_loader: train loader
            each_step_callback: (epoch, step, loss, pred, label)
        """
        tps : SoftmaxReModelParams.TrainParams = net.params.train_params
        loss_func = nn.CrossEntropyLoss()

        loss_func.to(device=device)
        net.to(device)

        no_decay = ["bias", "LayerNorm.weight"] # bias no decay, norm layer no deacy
        model_param = list(net.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if space[0] == 'bert_module' or space[0] == "bert":
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        grouped_params = [
            # bert params
            {
                "params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": tps.bert_decay, 
                'lr': tps.bert_lr
            },
            {
                "params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0, 
                'lr': tps.bert_lr
            },

            # softmax params
            {
                "params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": tps.linear_decay, 
                'lr': tps.linear_lr 
            },
            {
                "params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0, 
                'lr': tps.linear_lr 
            }
        ]
        
        total_step = tps.num_epochs * len(train_loader)
        optimizer = AdamW(grouped_params, lr=tps.linear_lr, eps=tps.eps)
        scheduler : LRScheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(tps.warm_up_proportion * total_step), num_training_steps=total_step)

        for epoch in range(1, tps.num_epochs + 1):
            iteration = tqdm(train_loader)
            for data, label in iteration:
                net.train()
                for key in data.keys():
                    data[key] = data[key].to(device)
                label = label.to(device)
                # print('input', data['input_ids'])
                output = net(data['input_ids'], data['attention_mask'], data['token_type_ids'])
                # print('output', output)
                loss = loss_func(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_value = loss.item()
                iteration.set_description('training(loss: %3.3f)'% loss_value)

                total_step = (epoch-1)*len(iteration) + iteration.n
                if each_step_callback:#(epoch, total_step, loss       , pred                           , label)
                    each_step_callback (epoch, total_step, loss.item(), argmax(output, dim=-1).tolist(), label.tolist())
    

    @staticmethod
    def valid(
        net,
        valid_loader,
        device = 'cuda:0',
    ) -> tuple[list[int], list[int]]:
        r""" 返回模型预测值 preds 和 目标值 targets，按照 (preds, targets) 返回 """
        loss_func = nn.CrossEntropyLoss()
        net.to(device)
        net.eval()
        iteration = tqdm(valid_loader)
        preds, targets = [], []
        with torch.no_grad():
            for data, label in iteration:
                for key in data.keys():
                    data[key] = data[key].to(device)
                label = label.to(device)

                output = net(data['input_ids'], data['attention_mask'], data['token_type_ids'])
                loss = loss_func(output, label)
                loss_value = loss.cpu().detach().item()
                iteration.set_description('validating(loss: %3.3f)'% loss_value)

                for i in range(output.shape[0]): # batch size loop
                    preds.append( argmax(output[i]).item() )
                    targets.append( label[i].item() )
            
        return preds, targets

