r"""
    此为控制保存模型、训练模型、加载模型的类
"""

from pathlib import Path
import json5
from torch import Tensor, argmax
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report as re_valid_report
from seqeval.metrics import classification_report as ner_valid_report

from .contruction_model_types import *


class KGConstructionController:
    r"""
    业务逻辑类型，输入只有文件路径，会按照每个文件夹应有的结构来执行业务逻辑

    应有结构长啥样后面再归纳，CONFIG_DIR 可能后续会去掉
    """

    def __init__(
            self,
            RELY_MODEL_DIR  : Path,
            SAVE_MODEL_DIR  : Path,
            DATASETS_DIR    : Path,
            CONFIG_DIR      : Path,
            ) -> None:
        r"""
        Args:
            `RELY_MODEL_DIR`: 依赖的外部模型的路径，目前是bert和albert
            `SAVE_MODEL_DIR`: 保存模型路径
            `DATASET_DIR`: 数据集文件夹所在路径
            `CONFIG_DIR`: 默认模型参数/配置文件夹所在路径
        """
        self.RELY_MODEL_DIR = RELY_MODEL_DIR 
        self.SAVE_MODEL_DIR = SAVE_MODEL_DIR 
        self.DATASETS_DIR   = DATASETS_DIR   
        self.CONFIG_DIR     = CONFIG_DIR   

        self.RE_SAVE_DIR = self.SAVE_MODEL_DIR.joinpath('re')
        self.NER_SAVE_DIR = self.SAVE_MODEL_DIR.joinpath('ner')

        self.BERT_DIR = self.RELY_MODEL_DIR.joinpath('bert')
        self.ALBERT_DIR = self.RELY_MODEL_DIR.joinpath('albert')

        self.ner_model_list: list[NerModelComposition] = []
        self.re_model_list: list[ReModelComposition] = []

    def add_ner(self, net, embedder = None)->None:
        r"""
        向现有激活ner模型列表中添加一个模型
        """
        self.ner_model_list.append(
            ReModelComposition(
                model=net,
                embedder=embedder,
            )
        )
        return len(self.ner_model_list) - 1

    def init_ner_from_params(self, params: NerModelParamTypes) -> int:

        if isinstance(params, BertBilstmNerModelParams):
            if not bool(params.hyper_params.num_labels):
                params.hyper_params.num_labels = len(self.get_dataset_ner_label_transer(params.dataset))
            net = BertBilstmNerModel(
                params=params, 
                bert_root_dir=self.BERT_DIR
            )
            embedder = BertBilstmNerEmbedder(
                bert_url= self.BERT_DIR.joinpath(params.hyper_params.bert),
                seq_len=params.hyper_params.seq_len
            )
            return self.add_ner(net, embedder=embedder)
        
        else:

            raise TypeError('NO SUCH NER TYPE PARAMS')
        
    def init_ner(self, type: NerTypes)->int:
        r"""
        从默认参数文件中加载ner模型
        """
        return self.init_ner_from_default_file(type)

    def init_ner_from_default_file(self, type: NerTypes)->int:
        r"""
        从默认参数文件中加载ner模型
        """
        PARAM_DIR = self.CONFIG_DIR.joinpath('ner_model_params')

        param_file_name, param_type = get_default_params_file_name(type)

        with open( PARAM_DIR.joinpath(param_file_name), 'r', encoding='UTF-8') as fp:
            params = param_type.from_dict( json5.load( fp=fp ))
  
        return self.init_ner_from_params(params)
        

    def add_re(self, net, embedder = None)->int:
        self.re_model_list.append(
            ReModelComposition(
                model=net,
                embedder=embedder,
            )
        )
        return len(self.re_model_list) - 1


    def init_re_from_default_file(self, type: ReTypes) -> int:
        
        PARAM_DIR = self.CONFIG_DIR.joinpath('re_model_params')

        param_file_names = {
            ReTypes.BERT_SOFTMAX: ('bert_softmax.json', SoftmaxReModelParams )
        }

        param_file_name, param_type = param_file_names[type]

        with open( PARAM_DIR.joinpath(param_file_name), 'r', encoding='UTF-8') as fp:
            params = param_type.from_dict( json5.load( fp=fp ))
        
        return self.init_re_from_params( params )


    def init_re(self, type: ReTypes) -> int:
        return self.init_re_from_default_file(type)
    

    def remove_ner(self, index: int) -> None:
        del self.ner_model_list[index]


    def remove_re(self, index: int) -> None:
        del self.re_model_list[index]


    def init_re_from_params(self, params: ReModelParamTypes):

        if isinstance(params, SoftmaxReModelParams):
            if not bool(params.hyper_params.num_labels):
                params.hyper_params.num_labels = len(self.get_dataset_re_label_transer(params.dataset))
            net = SoftmaxReModel(
                params=params, 
                bert_root_dir=self.BERT_DIR
            )
            embedder = SoftmaxReEmbedder(
                bert_url= self.BERT_DIR.joinpath(params.hyper_params.bert),
                seq_len=params.hyper_params.seq_len
            )
            return self.add_re(net, embedder=embedder)
        
        else:
            
            raise TypeError('NO SUCH RE TYPE PARAMS')
    
    @staticmethod
    def get_dataset_ner_label_transer(dataset_name: str) -> Union[CLNerLabelTranser, Any]:
        r"""
        根据数据集名称获得命名体识别标签id转换
        
        Args:
            `dataset_name`: 'Chinese-Literature-NER-RE-Dataset', 'dgre', 'DuIE2.0'
        """
        return dataset_ner_label_transer_mapping[dataset_name]()

    @staticmethod
    def get_dataset_re_label_transer(dataset_name: str) -> Union[DgreReLabelTranser, Any]:
        r"""
        根据数据集名称获得命名体识别标签id转换
        
        Args:
            `dataset_name`: 'Chinese-Literature-NER-RE-Dataset', 'dgre', 'DuIE2.0'
        """
        return dataset_re_label_transer_mapping[dataset_name]()

    def create_ner_dataset(self, composition: NerModelComposition, dataset_type: str) -> Dataset:
        r"""
        @TODO: maintenance this method for future development        
        Args:

        `composition`: 一个Ner模型实例的基本存储单位
        `dataset_dir: 数据集路径
        `dataset_type`：数据集类型，可选`'train'`,`'valid'`,`'test'`，有的数据集可能没有`test`数据集
        """
        model = composition.model
        dataset_path = self.DATASETS_DIR.joinpath(model.params.dataset) # datasets/$dataset
        if isinstance(model, BertBilstmNerModel):
            
            if model.params.dataset == 'Chinese-Literature-NER-RE-Dataset':
                mapping = {
                    'train': 'train.txt',
                    'valid': 'validation.txt',
                    'test': 'test.txt'
                }
                transer = self.get_dataset_ner_label_transer(model.params.dataset)
                ds = CLNerDataSet(
                    file_path=os.path.join(dataset_path, 'ner', mapping[dataset_type]),
                    encoder=NerCLToBertBilstmEncoder(composition.embedder, transer)
                )
                return ds
            # end dataset Chinese-Literature-NER-RE-Dataset
            
            elif model.params.dataset == 'dgre':
                mapping = {
                    'train': 'train.txt',
                    'valid': 'dev.txt'
                }
                transer = self.get_dataset_ner_label_transer(model.params.dataset)
                ds = DgreNerDataset(
                    file_path=os.path.join(dataset_path, 'ner_data', mapping[dataset_type]),
                    encoder=NerDgreToBertBilstmEncoder(composition.embedder, transer)
                )
                return ds
            # end dataset dgre
            
            elif model.params.dataset == 'DuIE2.0':
                mapping = {
                    'train':'duie_train.json',
                    'valid':'duie_dev.json',
                    'test':'duie_test2.json'
                }
                ds = DuIENerDataset(
                    file_path=dataset_path.joinpath(mapping[dataset_type]),
                    encoder=NerDuIEToBertBilstmEncoder(composition.embedder)
                )
                return ds
            else:
                raise NameError('no such dataset ner implemented')
        # end BertBilstmNerModel
        
        else:
            raise TypeError('unkown ner model type')


    def create_ner_train_loader(self, composition: NerModelComposition, num_workers: int) -> DataLoader:
        ds = self.create_ner_dataset(composition, 'train')
        return DataLoader(
            dataset=ds,
            batch_size=composition.model.params.train_params.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    def create_ner_valid_loader(self, composition: NerModelComposition, num_workers: int) -> DataLoader:
        ds = self.create_ner_dataset(composition, 'valid')
        return DataLoader(
            dataset=ds,
            batch_size=composition.model.params.train_params.batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    def create_re_dataset(self, composition: ReModelComposition, dataset_type: str) -> Dataset:
        model = composition.model
        dataset_name = model.params.dataset
        dataset_path = self.DATASETS_DIR.joinpath(model.params.dataset)
        if isinstance(model, SoftmaxReModel):

            if dataset_name == 'dgre':
                mapping = {
                    'train': 'train.txt',
                    'valid': 'dev.txt'
                }
                return DgreReDataset( 
                    dataset_path.joinpath('re_data', mapping[dataset_type] ), 
                    encoder=ReDgreToSoftmaxEncoder(
                        embedder=composition.embedder, 
                        label_transer=self.get_dataset_re_label_transer(dataset_name)
                    )
                )
            # end dataset dgre

            if dataset_name == 'DuIE2.0':
                mapping = {
                    'train': 'duie_train.json',
                    'valid': 'duie_dev.json',
                    'test': 'duie_test2.json'
                }
                return DuIEReDataSet(
                    dataset_path.joinpath(mapping[dataset_type]),
                    encoder = ReDuIEToBertSoftmaxEncoder(embedder=composition.embedder)
                )
            # end dataset duie

            else:
                raise NameError('no such dataset ner implemented')

        #end SoftmaxReModel

    def create_re_train_loader(self, composition: ReModelComposition, num_workers = 0) -> DataLoader:
        ds = self.create_re_dataset(composition, 'train')
        return DataLoader(
            dataset=ds,
            batch_size=composition.model.params.train_params.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    
    def create_re_valid_loader(self, composition: ReModelComposition, num_workers = 0) -> DataLoader:
        ds = self.create_re_dataset(composition, 'valid')
        return DataLoader(
            dataset=ds,
            batch_size=composition.model.params.train_params.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    @staticmethod
    def get_dataset_ner_label_transer(dataset_name: str):
        return dataset_ner_label_transer_mapping[dataset_name]()

    
    def ner_predicate(self, index: int, sentence: str)->list[str]:
        composition = self.ner_model_list[index]

        if isinstance(composition.model, NerTypes.BERT_BILSTM_CRF.value):
            transer = self.get_dataset_ner_label_transer(composition.model.params.dataset)
            model_in = composition.embedder(sentence, True)
            device = next(composition.model.parameters()).device
            # 转移到与model相同的设备中去
            for key in model_in.keys():
                model_in[key] = model_in[key].to(device)

            r, _ = composition.model(
                input_ids=model_in['input_ids'],
                attention_mask=model_in['attention_mask'].bool(),
                token_type_ids=model_in['token_type_ids']
                )
            if isinstance(r, Tensor):
                r = r.tolist()
            return transer.id2label(r[0])[1:]
        # elif ...
        else:
            raise TypeError('unkown ner model type')
        # get data set label transer
    
    def re_predicate(self, index, sentence: str, subject: str, object: str) -> str:
        composition = self.re_model_list[index]

        model = composition.model

        if isinstance(model, ReTypes.BERT_SOFTMAX.value):
            transer = self.get_dataset_re_label_transer(model.params.dataset)
            model_in = composition.embedder(sentence, subject, object, True)
            device = next(model.parameters()).device
            # 转移到与model相同的设备中去
            for key in model_in.keys():
                model_in[key] = model_in[key].to(device)
            pred = argmax(model.forward(
                input_ids=model_in['input_ids'],
                attention_mask=model_in['attention_mask'].bool(),
                token_type_ids=model_in['token_type_ids']
            )).item()
            return transer.id2label(pred)
        else:
            raise TypeError('unkown ner model type')

    def train_ner(self, index: int, device = 'cuda:0', num_workers: int = 0) -> None:
        r"""
        训练已加载的模型
        """
        composition = self.ner_model_list[index]
        train_loader = self.create_ner_train_loader(composition, num_workers=num_workers)
        self.before_ner_train(composition, train_loader)
        
        composition.model.train_epochs(
            composition.model,
            train_loader,
            device=device,
            each_step_callback=self.train_ner_handler
        )

        # self.after_ner_train()

    def valid_ner(self, index: int, device = 'cuda:0', num_workers: int = 0, output_dict = False) -> Union[str, dict]:
        composition = self.ner_model_list[index]
        valid_loader = self.create_ner_valid_loader(composition, num_workers=num_workers)
        
        preds, targets = composition.model.valid(
            composition.model,
            valid_loader,
            device=device
        )

        transer = self.get_dataset_ner_label_transer(composition.model.params.dataset)

        pred_labels, target_labels = [], []
        for pred in preds:
            pred_labels.append(transer.id2label(pred))
        for target in targets:
            target_labels.append(transer.id2label(target))
        
        print(ner_valid_report(target_labels, pred_labels, output_dict=True))
        # composition.model.set_report(json5.dumps(ner_valid_report(target_labels, pred_labels, output_dict=True)))

        return ner_valid_report(target_labels, pred_labels, output_dict=output_dict)


    def valid_re(self, index: int, device = 'cuda:0', num_workers: int = 0, output_dict = False) -> Union[str, dict]:
        composition = self.re_model_list[index]
        
        valid_loader = self.create_re_valid_loader(composition, num_workers=num_workers)
        
        preds, targets = composition.model.valid(
            composition.model,
            valid_loader,
            device=device
        )

        transer = self.get_dataset_re_label_transer(composition.model.params.dataset)

        targets = transer.id2label(targets)
        preds =  transer.id2label(preds)

        return re_valid_report(targets, preds, output_dict=output_dict)
        

    def train_re(self, index: int = 0, device = 'cuda:0', num_workers: int = 0) -> None:
        r"""
        训练已加载的模型
        """
        composition = self.re_model_list[index]
        train_loader = self.create_re_train_loader(composition, num_workers=num_workers)
        self.before_re_train(composition, train_loader)
        composition.model.train_epochs(
            composition.model,
            train_loader,
            device=device,
            each_step_callback=self.train_ner_handler
        )


    def save_ner_model(self, index: int) -> None:
        composition = self.ner_model_list[index]
        name = composition.model.params.name
        model = composition.model
        save_name = name + '.' + model.__class__.__name__
        save_dir = self.NER_SAVE_DIR.joinpath(save_name)
        model.save(save_dir)


    def save_re_model(self, index) -> None:
        composition = self.re_model_list[index]
        name = composition.model.params.name
        save_name = name + '.' + type(composition.model).__name__
        save_dir = self.RE_SAVE_DIR.joinpath(save_name)
        composition.model.save(save_dir)


    def load_ner_model(self, model_name: str) -> None:
        for i, c in enumerate(model_name[::-1]):
            if c == '.':
                name = model_name[0:i]
                type_name = model_name[i+1:]

        if type_name == 'BertBilstmNerModel':
            model = BertBilstmNerModel.load( self.NER_SAVE_DIR.joinpath(model_name) , self.BERT_DIR)
            model.params.name = name
            params = model.params
            embedder = BertBilstmNerEmbedder(
                bert_url= self.BERT_DIR.joinpath(params.hyper_params.bert),
                seq_len=params.hyper_params.seq_len
            )
            self.add_ner(model, embedder=embedder)
            return


    def load_re_model(self, model_name: str) -> int:
        r"""
        名字有.XXXX后缀？那就对了
        """
        name_list = model_name.split('.')
        name = ''.join(name_list[:-1])
        type_name = name_list[-1]
        print('loading re model %s(%s)'%(name, type_name))
        if type_name == 'SoftmaxReModel':
            model = SoftmaxReModel.load( self.RE_SAVE_DIR.joinpath(model_name), self.BERT_DIR )
            params = model.params
            embedder = SoftmaxReEmbedder(
                bert_url= self.BERT_DIR.joinpath(params.hyper_params.bert),
                seq_len=params.hyper_params.seq_len
            )
            self.add_re(model, embedder=embedder)
            return len(self.re_model_list) - 1
    

    def ner_re_joint_predicate(self, sentence: str, ner_index: int, re_index: int) -> tuple[list[NerEntity], list[Relation]]:
        ner_labels = self.ner_predicate(ner_index, sentence)
        entity_array = convert_label_seq_to_entity_pos(sentence, ner_labels)

        relations = []
        for i, subject in enumerate(entity_array):
            for object in entity_array[i+1:]:
                rel = self.re_predicate(re_index, sentence, subject.entity, object.entity)
                relations.append( subject, rel, object )
        entity_array = entity_array[::-1]

        entity_array_reverse = entity_array.reverse()
        for i, subject in enumerate(entity_array_reverse):
            for object in entity_array_reverse[i+1:]:
                rel = self.re_predicate(re_index, sentence, subject.entity, object.entity)
                relations.append( subject, rel, object )
        entity_array = entity_array[::-1]

        return entity_array, relations
            
        
    # begin implement needed APIs ----------------------------------------------------------------------------------------------------------------------

    def train_ner_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]) -> None:
        return


    def train_re_handler(self, epoch: int, total_step: int, loss: float, pred: list[int], label: list[int]) -> None:
        return
    

    def before_ner_train(self, composition: NerModelComposition, loader: DataLoader) -> None:
        return
    

    def before_re_train(self, composition: ReModelComposition, loader: DataLoader) -> None:
        return


        
