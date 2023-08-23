from torch.utils.data import Dataset
from typing import Callable
from pybrat.parser import BratParser, Example



class ReDataSet(Dataset):
    def __init__(self, 
            encoder: Callable[[str, str, str], dict], 
            sentence_file_path: str,  
            annotation_file_path: str
        ) -> None:
        r""" encoder: (sentence, subject, object)-> { 'input_ids': Tensor, 'attention_mask': Tensor, 'token_type_ids': Tensor } """
        super().__init__()
        self.encoder = encoder
        self.parser = BratParser()
        self.parser.parse()
    
    def __getitem__(self, index) -> None: