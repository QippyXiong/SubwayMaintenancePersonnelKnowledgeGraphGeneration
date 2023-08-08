这是一个测试项目，用ChineseLiterature测试知识图谱生成

data_loader.py : 
ner_model : 

#### ChineseLiterature数据形式

[点此访问数据集](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)

数据集下载放到data/dataset目录下

### NER 模型

#### 参数化训练
为了测试一系列参数（如使用哪个预训练模型，多少层，多大）在数据集上的效果，完成了一个参数化的ner模型

通过修改 config/ner_config.json 文件中的参数来尝试训练一个固定参数的ner

可以将训练好的模型保存在 model/ner 文件夹中，保存结果将包括：

    $(模型名称):
    - ner_model.bin : pytorch模型文件
    - config.json : 此模型初始化、训练的参数
    - description.txt : 此模型的一段描述
    - report.txt : 此模型在validation数据集上的测试结果

后两个文件可以缺省

通过编写运行 src/main.py 来完成，一个例子是：

```python
from controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.init_ner() # 初始化ner模型，embedder，参数为 config/ner_config.json 中的参数
    controller.ner.name = 'ner-albert-bilstm-crf_v0.1' # 模型的名字，保存的时候文件夹名字就是这个
    controller.ner.description = '尝试albert微调' # 试过了，结果不太好
    controller.train_ner() # 训练模型
    controller.ner.save() # 保存模型，这个保存没有报告
    controller.validate_ner() # 验证模型
    controller.ner.save() # 保存模型，会有报告文件
    print( controller.ner_task('我爸是李刚') ) # 做一次实体识别
```

如果要加载已训练好的模型，一个例子是：

```python
from controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.load_ner('ner-bilstm-crf_v0.2')
    print( controller.ner_task('我爸是李刚') )
```
