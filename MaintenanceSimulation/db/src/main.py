from Neo4jDBMS.datatype import *
from Neo4jDBMS import DataBaseManager
r"""
    写了半天DBMS发现没啥好写的，很多操作都不能越过neo4j Cypher语句
    唯一可能有点意义的就是结构化表意了
"""

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




