import os
from transformers import AutoTokenizer, BertForMaskedLM, logging as trans_loger
from data_loader import NerDataSet, DATASET_DIR, cn_bert_tokenizer
from torch.utils.data import DataLoader
from torch import Tensor
from ner_model import NerModel

a = NerModel()


