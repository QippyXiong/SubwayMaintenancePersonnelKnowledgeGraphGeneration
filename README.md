# SubwayMaintenancePersonnelKnowledgeGraphGeneration
简称SMPKG

#### ChineseLiteratureKG
第一个子项目，ChineseLiteratureKG，完成其内部的ner和re代码和KG设计
ChineseLiteratureKG使用Chinese Literature数据集，你需要将数据集clone到data/dataset目录下
数据集地址：[点此访问数据集](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)

#### MaintenanceSimulation

目前已经完成了Neo4jDBMS中DATABASE manager大体功能执行的编写

你可以通过创建DataManager来连接neo4j服务器

如果想在ChineseLiteratureKG中使用Neo4jDBMS，可以使用如下代码
```python
import sys
import os

# current_dir 为代码文件所在文件夹
current_dir = os.path.dirname(__file__)
# 不应使用 .或者 ..来表示路径，容易导致代码兼容性差，因为python默认的相对目录是命令行调用python所在的目录，锁定文件目录以避免文件路径定位问题
# project_dir应为文件夹ChineseLiterature，可以通过调节current_dir.split(os.path.sep)[:-1]中的-1来改变相对路径，比如-2就是上两级，-3是上三级，以此类推，此处等效于..，即此代码文件所在文件夹的上一层文件夹
project_dir = ''.join([ item + os.path.sep for item in current_dir.split(os.path.sep)[:-1]]) # ..

DBMS_module_path = os.path.join(project_dir, "MaintenanceSimulation", "db", "src")
sys.path.append(DBMS_module_path) # 添加模块路径
print(DBMS_module_path) # 查看一下是不是正确路径
from Neo4jDBMS import DataBaseManager # DataBaseManager即为管理类
from neo4j import Record
```
`DataDataBaseManager`通过`execute_query`方法向`neo4j`数据库传送`Cypher`指令
```python
manager = DataBaseManager()

# 搜索所有结点，并返回
records, summary, keys = manager.execute_query(
    f'''
        MATCH (n) RETURN n
    '''
)
# 访问搜索结果
for record in records:
    node = record['n']
    for key in node.keys(): # 类似于dict，node.keys()包含了所有键值，如果你已知道key的值，可以直接访问
        print(f'{key}:{ node[key] }')

```

也可以直接通过`manager`创建结点
```python
from Neo4jDBMS.datatype import * # 导入必要数据类型
# 构造manager，连接数据库
manager = DataBaseManager()
# 创造一个结点
handsome_boy = Entity()
# 设置neo4j中结点类型为帅哥
handsome_boy.class_ = Class("帅哥") # 不可直接赋值为 "帅哥"
# 为这个结点添加属性
handsome_boy.props["名字"] = "平如愿"
handsome_boy.props["外貌"] = "帅啊"
# 在数据库中创建这个结点
manager.createEntity(handsome_boy)

# 关闭连接
manager.close()
```

还没写好创建关系的方法，要使用cypher的话，一个参考是：

```python
query = f'''
            MATCH (a:{ class_name1 }), (b:{ class_name2 })
            WHERE a.{ props_key1 } = "{ props_value1 }" AND b.{ props_key2 } = "{ props_value2 }"
            CREATE (a)-[:{ forward_relation }]->(b)
            CREATE (b)-[:{ backward_relation }]->(a)
        '''
        # 可以只有一个CREATE或添加更多行CREATE
manager.execute_query(query)
```
