from transformers import AutoTokenizer
from torch import tensor, long
from typing import Optional, Union
from os import path

current_path = path.dirname(__file__)
project_path = ''.join([ item + path.sep for item in current_path.split(path.sep)[:-3]]) # ../../..
MODEL_DIR = path.join(project_path, 'ChineseLiteratureKG', 'model')
CN_BERT_DIR = path.join(project_path, 'ChineseLiteratureKG', 'model', 'chinese-bert-wwm-ext')

class BertEmbedder(object):
    r""" 用于将单个句子处理成输入数据 """
    def __init__(self, bert_url: str = CN_BERT_DIR, seq_len: Optional[int] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(bert_url)
        self.seq_len = seq_len
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.pad_token = pad_token

    def __call__(self, sentence: str, batch: bool = False, seq_len: Optional[int] = None) -> dict:
        r""" 非 data_loader 加载的时候（单个句子判断），设置 batch 为 True """
        if seq_len is None: seq_len = self.seq_len
        # perData = self.tokenizer(sentence, return_offsets_mapping=True, max_length=self.seq_len, truncation=True)
        perData = self.tokenizer(sentence, padding='max_length', max_length=seq_len, truncation=True)
        # self.data.append({ key: Tensor(perData[key]) for key in perData })
        result = {}
        for key in perData:
            if len(perData[key]) > seq_len:
                print(f'err: { sentence }')
                assert(True)
            if batch:
                result[key] = tensor([perData[key]], dtype=long)
            else:
                result[key] = tensor(perData[key], dtype=long)
    
        return result

        input_ids      = tensor(perData['input_ids']     , dtype=long)
        token_type_ids = tensor(perData['token_type_ids'], dtype=long)
        attention_mask = tensor(perData['attention_mask'], dtype=long).bool()
        # padding
        input_ids       += [0] * padding_len
        token_type_ids  += [0] * padding_len
        attention_mask  += [0] * padding_len
    
    def decode(self, data: Union[list[int], dict]) -> str:
        r""" 此方法意义不大，建议直接调用 .tokenizer.convert_ids_to_tokens """
        if type(data) is list:
            return self.tokenizer.convert_ids_to_tokens(data)
        elif type(data) is dict:
            return self.tokenizer.convert_ids_to_tokens(data['input_ids'])