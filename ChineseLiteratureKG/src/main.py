import os
from transformers import PreTrainedTokenizerFast, BertForMaskedLM, logging as trans_loger
from utils.DataSet import NerDataSet, NER_DATASET_DIR, CN_BERT_DIR
from torch.utils.data import DataLoader
from torch import Tensor
from ner_model import NerModel

# cn_bert_tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(CN_BERT_DIR)
# ds = NerDataSet(os.path.join(NER_DATASET_DIR, "train.txt"), 128, tokenizer=cn_bert_tokenizer)

# print(ds[0][0])
# for key in ds[0][0]:
#     print(ds[0][0][key].shape)
# print( cn_bert_tokenizer.convert_ids_to_tokens(ds[0][0]['input_ids']) )
#     # print(label)

import sys
import os
current_dir = os.path.dirname(__file__)
project_dir = ''.join([ item + os.path.sep for item in current_dir.split(os.path.sep)[:-2]]) # ../..
DBMS_module_path = os.path.join(project_dir, "MaintenanceSimulation", "db", "src")
sys.path.append(DBMS_module_path) # 添加模块路径
print(DBMS_module_path)
from Neo4jDBMS import DataBaseManager
from neo4j import Record

manager = DataBaseManager()
records, summary, keys = manager.execute_query(
    f'''
        MATCH (n) RETURN n
    '''
)
for record in records:
    node = record['n']
    for key in node.keys():
        print(f'{key}:{ node[key] }')

from Neo4jDBMS.datatype import *

