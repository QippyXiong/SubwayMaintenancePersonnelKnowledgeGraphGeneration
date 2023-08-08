from ner_model import NerModel, load_ner_model
from ner_model.Train import train_ner, validate_ner
import os
import json5 as json # 使用json5以保证config文件内可以写注释
from utils.DataSet import NerDataSet, NER_DATASET_DIR, NerLabelTranser
from torch.utils.data import DataLoader
from utils.BertEmbedder import BertEmbedder, MODEL_DIR

class Controller:
    r""" 
        此类综合控制 ner, re, db 实现业务逻辑
        原因是在main函数中写业务逻辑时发现太长了，不太好
        写个run文件又不好后续更改集成，干脆写个Controller类

        你会发现这个类的函数大多没有参数，没错，这个类就是写死的业务逻辑类

        属性：
            ner: 
            re:
    """

    def __init__(self) -> None:
        r"""

        """
        self.ner_label_transer = NerLabelTranser(NerLabelTranser.CONFIG_FILE_PATH)
    
    current_dir = os.path.dirname(__file__)
    PROJ_DIR = ''.join([ item + os.path.sep for item in current_dir.split(os.path.sep)[:-2]])
    CONFIG_DIR = os.path.join(PROJ_DIR, 'config')
    
    def init_ner(self)-> None:
        r""" 根据config中ner_config.json 初始化 ner 模型，并初始化 bert embedder """
        ner_config = {}
        with open(os.path.join(Controller.CONFIG_DIR, 'ner_config.json'), 'r', encoding='UTF-8') as fp:
            ner_config = json.load(fp)
        self.ner = NerModel(ner_config)
        if 'bert' in ner_config['model_parms'].keys():
            self.bert_embedder = BertEmbedder( os.path.join(MODEL_DIR, ner_config['model_parms']['bert'] ) )
        else: # 默认是 chinese-bert-wwm-ext
            self.bert_embedder = BertEmbedder()
        
    def load_ner(self, name: str):
        self.ner = load_ner_model(name)
        if 'bert' in self.ner.config['model_parms'].keys():
            self.bert_embedder = BertEmbedder( os.path.join(MODEL_DIR, self.ner.config['model_parms']['bert'] ) )
        else: # 默认是 chinese-bert-wwm-ext
            self.bert_embedder = BertEmbedder()

    def train_ner(self):
        self.bert_embedder.seq_len = self.ner.seq_len
        train_ds = NerDataSet( os.path.join(NER_DATASET_DIR, "train.txt"), self.bert_embedder, transer=self.ner_label_transer) 
        # valid_ds = NerDataSet( os.path.join(NER_DATASET_DIR, "validation.txt"), self.bert_embedder, self.ner_label_transer)
        # valid_loader = DataLoader(valid_ds, batch_size=self.ner.config['train_config']['batch_size'], shuffle=False)
        if train_ds.num_label() != self.ner.config['model_parms']['num_labels']:
            print(f'模型与数据集参数不相同:(model.num_labels: { self.ner.config["model_parms"]["num_labels"] } while { train_ds.num_label() })')
        train_loader = DataLoader(train_ds, batch_size=self.ner.config['train_config']['batch_size'], shuffle=True)
        train_ner(self.ner, train_loader, self.ner.config['train_config']['epochs'], None)
        
    def validate_ner(self)-> dict:
        r""" 会返回验证结果字典 """
        self.bert_embedder.seq_len = self.ner.seq_len
        valid_ds = NerDataSet( os.path.join(NER_DATASET_DIR, "validation.txt"), self.bert_embedder, self.ner_label_transer)
        valid_loader = DataLoader(valid_ds, batch_size=self.ner.config['train_config']['batch_size'], shuffle=False)
        report, report_text = validate_ner(self.ner, valid_loader)
        self.ner.valid_report = report_text
        self.ner.report['validation'] = report
        print(report_text)
        return report
    
    def ner_task(self, sentence: str):
        r""" 用 ner 模型作一次实体识别，返回标签序列 """
        self.bert_embedder.seq_len = self.ner.seq_len # 两个要相同，不然出事了
        r, _ = self.ner( self.bert_embedder(sentence, batch=True) )

        # r[0]是由于batch_size=1，1:-1是除去开头[cls]和[sep]
        return self.ner_label_transer.id2label( r[0][1:-1] ) 

    def init_re(self):
        ...