import torch
import numpy as np
from torch.utils.data import Dataset


# 读取数据
def read_data(file_path):
    # 读取数据集
    with open(file_path, "r", encoding="utf-8") as f:
        # 按行读取并去除前后空格
        content = [line.strip() for line in f.readlines()]

    # 读取空行行号
    index = [-1]
    index.extend([i for i, line in enumerate(content) if ' ' not in line])
    index.append(len(content))

    # 按空行分割句子，读取sentence和tags
    sentences, tags = [], []
    for i in range(len(index) - 1):
        sentence, tag = [], []
        segment = content[index[i] + 1:index[i + 1]]
        for line in segment:
            sentence.append(line.split()[0])
            tag.append(line.split()[1])
        sentences.append(''.join(sentence))
        tags.append(tag)

    # 去除空的句子及标注序列，一般放在末尾
    sentences = [_ for _ in sentences if _]
    tags = [_ for _ in tags if _]

    return sentences, tags

# 检查标签是否正确标注
# def label2id(train_file_path):
#
#     train_sents, train_tags = read_data(train_file_path)
#
#     # 标签转换成id，并保存成文件
#     unique_tags = []
#     for seq in train_tags:
#         for _ in seq:
#             if _ not in unique_tags:
#                 unique_tags.append(_)
#     return unique_tags


class NerDataset(Dataset):
    def __init__(self, file_path, args, tokenizer):
        self.text, self.labels = read_data(file_path)
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len
        # print(label2id(file_path))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = [_ for _ in self.text[item]]
        labels = self.labels[item]
        # 截断
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        # 填充
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # print(item, len(data["input_ids"]), len(data["attention_mask"]), len(data["labels"]))
        return data


