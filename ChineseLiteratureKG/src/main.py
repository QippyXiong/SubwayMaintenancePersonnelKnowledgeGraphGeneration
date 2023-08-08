from controller import Controller
from utils.BertEmbedder import BertEmbedder

if __name__ == '__main__':
    con = Controller()
    con.load_ner('ner-bilstm-crf_v0.1')
    con.ner.description = "测试biltsm-crf模型1"
    con.ner.save()