from .__init__ import NerModel
from utils.animator import Animator
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import AdamW
from torch.nn import Parameter
from torch import Tensor
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from typing import *
from seqeval.metrics import f1_score, accuracy_score, classification_report
from utils.DataSet import NerLabelTranser

from threading import Thread

def validate_ner(
    net: NerModel, 
    validation_loader: DataLoader,
    device = 'cuda',
)-> tuple[float, float]:
    r""" return f1 score, accuracy """
    net.to(device)
    net.eval()
    validate_label_true = []
    validate_label_pred = []
    batch_size = validation_loader.batch_size
    iter = tqdm(validation_loader)
    transer : NerLabelTranser = validation_loader.dataset.transer
    for data, label in iter:
        for key in data.keys():
            data[key] = data[key].to(device)
        label = label.to(device)
        logit, loss = net(data, label)
        loss_value = loss.detach().cpu().item()
        iter.set_description('validation(loss: %3.3f)'%loss_value)
        batch_size = data['attention_mask'].shape[0]
        for i in range(batch_size):
            length = sum(data['attention_mask'][i]).item()
            validate_label_true.append(
                transer.id2label(label[i][1:length-1].detach().cpu().numpy())
            )
            validate_label_pred.append(
                transer.id2label(logit[i][1:length-1])
            )

    valid_f1 = f1_score(validate_label_true, validate_label_pred)
    valid_acc = accuracy_score(validate_label_true, validate_label_pred)
    report = classification_report(validate_label_true, validate_label_pred)
    for i in range(len(report)):
        print(report[i])
    return valid_f1, valid_acc, report


def train_ner_model_epoch(
        net: NerModel, 
        train_loader: DataLoader,
        optimizer: AdamW,
        scheduler: LRScheduler,
        device = 'cuda',
        animator: Optional[Animator] = None
    ) -> None:
    r"""
        ner(命名实体识别训练函数，训练一个epoch)
        net: 神经网络
        train_loader: 训练集加载迭代器
        epochs: 训练集总遍历次数
        training_config: dict :
            {
                'warm_up_proportion': float,
                'bert_lr'           : float,
                'bert_decay'        : float,
                'crf_lr'            : float,
                'crf_decay'         : float,
                'other_lr'          : float,
                'other_decay'       : float,
                'eps'               : float,
            }
        device: 设备
        animator: utils.Animator，使用曲线图显示模型准确度变化
        validation_loader: 验证集迭代器
    """

    net.to(device)

    net.train()
    loss_value = 0.0
    iter = tqdm(train_loader)
    for data, label in iter:
        iter.set_description('training(loss: %3.3f)'%loss_value)
        for key in data.keys():
            data[key] = data[key].to(device)
        label = label.to(device)
        logit, loss = net(data, label)
        loss_value = loss.cpu().detach().item()
        # @TODO: 添加梯度积累方法
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if animator is not None:
            animator.add(iter.n, loss_value)

def get_transformer_optimizer(
        named_params : Iterator[Tuple[str, Parameter]],
        total_step: int,
        warm_up_proportion: float,
        bert_lr: float,
        bert_decay: float,
        crf_lr: float,
        crf_decay: float,
        other_lr: float,
        other_decay: float = 0.,
        eps: float = 1e-8,
    ) -> tuple[AdamW, LRScheduler]:
    
    bert = ('bert')
    crf = ('crf')
    # 无weight_decay的参数
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight')

    # 将参数分为3类，可以按照不同的学习率
    grouped_parms = [
        { # bert层                                           # 名字中包含bert                  # 名字中不包含no_decay中的字符串
            'params': [parm for name, parm in named_params if any(n in name for n in bert) and not any(n in name for n in no_decay)],
            'lr': bert_lr,
            'weight_decay': bert_decay,
        }
        ,
        { # crf层                                            # 名字中包含crf                   # 名字中不包含no_decay中的字符串
            'params': [parm for name, parm in named_params if any(n in name for n in crf ) and not any(n in name for n in no_decay)],
            'weight_decay': crf_decay,
            "lr": crf_lr
        }
        ,
        { # 其他层（不在crf中，不在bert中，不在no_decay中）这里应该是bilstm
            'params': [parm for name, parm in named_params if not any(n in name for n in crf ) and not any(n in name for n in no_decay) and not any(n in name for n in bert)],
            'weight_decay': other_decay,
            'lr': other_lr,
        }
        ,
        { # no decay层
            'params': [parm for name, parm in named_params if any(n in name for n in no_decay)],
            'weight_decay': 0.,
            'lr': other_lr
        }
    ]

    optimizer = AdamW(grouped_parms, lr=other_lr, eps=eps)

    # 线性warmup，保持模型训练时的稳定
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warm_up_proportion * total_step), total_step)
    return optimizer, scheduler

def get_optimizer_scheduler(named_params: Iterator[Tuple[str, Parameter]], total_step, train_config: dict)->tuple[AdamW, LRScheduler]:
    return get_transformer_optimizer(
        named_params = named_params,
        total_step = total_step,
        # training config
        warm_up_proportion  = train_config[ 'warm_up_proportion' ],
        bert_lr             = train_config[ 'bert_lr'            ],
        bert_decay          = train_config[ 'bert_decay'         ],
        crf_lr              = train_config[ 'crf_lr'             ],
        crf_decay           = train_config[ 'crf_decay'          ],
        other_lr            = train_config[ 'other_lr'           ],
        other_decay         = train_config[ 'other_decay'        ],
        eps                 = train_config[ 'eps'                ],
    )

def train_ner(
        net: NerModel, 
        train_loader: DataLoader, 
        epochs: int,
        validation_loader : Optional[DataLoader] = None,
        device = 'cuda'
    ) -> None:
    r"""
        训练ner模型，并使用 utils.Animator 显示每个epoch结果
        net: 神经网络
        train_loader: 训练集加载迭代器
        epochs: 训练集总遍历次数
        training_config: dict :
            {
                'warm_up_proportion': float,
                'bert_lr'           : float,
                'bert_decay'        : float,
                'crf_lr'            : float,
                'crf_decay'         : float,
                'other_lr'          : float,
                'other_decay'       : float,
                'eps'               : float,
            }
        device: 设备
        validation_loader: 验证集迭代器
    """
    lengends = []
    animator = object()
    if epochs != 1:
        lengends = ['train f1', 'train acc']
        if validation_loader:
            lengends += ['valid f1', 'valid acc']
        animator = Animator('epoch', 'acc/f1', lengends, [1, epochs], [0., 1.])
    else:
        lengends = ['loss']
        animator = Animator('step', 'loss', lengends, [1, len(train_loader)])

    optimizer, scheduler = get_optimizer_scheduler(
        named_params = net.named_parameters(),
        total_step = epochs*len(train_loader),
        train_config= net.config['train_config']
    )

    def _task():
        # 如果只有1个epoch，直接显示loss曲线
        for epoch in range(1, epochs+1):
            train_ner_model_epoch(net, train_loader, optimizer, scheduler, device, animator if epochs == 1 else None)
            if validation_loader is not None: # 在验证集上作一次检查
                valid_f1, valid_acc = validate_ner(net, validation_loader)
                print(f'[epoch:{ epoch }]\n\t[valid]f1 score: { valid_f1 }, accuracy: { valid_acc }')
                if epochs != 1: # 不止训练一个epoch，按epoch-(f1,accuracy)显示
                    y_point = [ valid_f1, valid_acc ]
                    animator.add(epoch, y_point)
    
    t = Thread(target = _task)
    t.start()
    animator.show()
    t.join()
