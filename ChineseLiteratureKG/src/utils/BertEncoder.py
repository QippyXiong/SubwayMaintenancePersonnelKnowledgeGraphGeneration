from transformers import PreTrainedTokenizerFast
from torch import Tensor


class BertEncoder(object):
    
    def __init__(self, tokenizer: PreTrainedTokenizerFast, seq_len: int = None) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # self.pad_token = pad_token

    def __call__(self, sentence: str) -> dict:
        # perData = self.tokenizer(sentence, return_offsets_mapping=True, max_length=self.seq_len, truncation=True)
        perData = self.tokenizer(sentence, max_length=self.seq_len, truncation=True)
        # self.data.append({ key: Tensor(perData[key]) for key in perData })
        result = {}
        for key in perData:
            item = perData[key]
            if self.seq_len: # padding
                item += [0] * (self.seq_len - len(item))
            result[key] = Tensor(item)

        return result

        input_ids = perData['input_ids']
        token_type_ids = perData['token_type_ids']
        attention_mask = perData['attention_mask']
        # padding
        input_ids       += [0] * padding_len
        token_type_ids  += [0] * padding_len
        attention_mask  += [0] * padding_len