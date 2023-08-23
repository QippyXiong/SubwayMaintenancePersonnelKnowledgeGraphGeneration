from controller import Controller 
from pybrat.parser import BratParser
from utils.DataSet import NER_DATASET_DIR
from pathlib import Path

def main():
    print(__file__)
    re_train_path = Path(NER_DATASET_DIR).resolve().parent
    re_train_path = re_train_path.joinpath('relation_extraction', 'Validation')
    print(re_train_path)
    parser = BratParser()
    examples = parser.parse(re_train_path)
    print(len(examples))
    for example in examples:
        print(example)
    

    return
    tokenizer = AutoTokenizer.from_pretrained(CN_BERT_DIR)

    sentence = "叔叔我啊，最喜欢[SEP]变形金刚[SEP]了"
    data = tokenizer(sentence)
    print(len(sentence), len(data['input_ids']))
    print(tokenizer.convert_ids_to_tokens(data['input_ids']))

    """
    con = Controller()
    con.init_ner()
    con.ner.name = 'albert-bilstm-crfv0.2'
    con.train_ner()
    con.validate_ner()
    con.ner.save()
    """

if __name__ == '__main__':
    main()