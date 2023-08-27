r"""
    这个包包含所有实体识别模型的实现
"""

# 转义名字
from .bert_bilstm_ner_model import NerModel as BertBilstmNerModel
from .bert_bilstm_ner_model import NerModelParams as BertBilstmNerModelParams
from .bert_bilstm_ner_model import NerEmbedder as BertBilstmNerEmbedder