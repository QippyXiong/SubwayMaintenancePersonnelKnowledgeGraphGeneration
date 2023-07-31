import os
from neo4j import GraphDatabase, Driver
from datatype import *
from typing import *
from statement import Statement

current_dir = os.path.dirname(__file__)
# ../..
db_dir = ''.join([ folder + os.sep for folder in current_dir.split(os.sep)[:-2] ])

class DataBaseManager():
    r"""
        For neo4j db connection and base ctrl
        you can acess crud by create, search, update, delete method
        if you want to call basic query, use self.driver
        use close function to close connection
    """
    def __init__(self, info : dict) -> None:
        r"""
            info dict is like:
            {
                "uri": "neo4j://localhost",
                "auth": ["username", "password"]
            }
            same as the db.json content
        """
        self.driver : Driver = GraphDatabase.driver(info["uri"], auth=tuple(info["auth"]))
        self.driver.verify_connectivity()

    
    def close(self) -> None:
        r""" close connection to database """
        self.driver.close()


    # crud begin @ToDo: compelete CRUD
    def create(self, item : Union[Union[Node, Edge], list[Union[Node, Edge]]]) -> Statement:
        r""" return a simple create a(n) node/entity or edge/relation statement """
        if type(item) is Iterable: # 添加一系列结点或边
            query_str = 'CREATE '.join([ str(i) + ',' for i in item ])
            return Statement(self.driver, query_str[:-1]) # 去掉最后的','
        elif type(item) is Node or type(item) is Edge: # 添加一个边
            return Statement(self.driver, f'CREATE {item}')
        else: # 显然我们不能创造除结点和边之外的东西
            raise TypeError("You can only create node/entity or edge/relation")

    def search(self) -> Statement:
        r""" retrieve """
        ...
    
    def delete(self) -> Statement:
        ...
    
    def update(self) -> Statement:
        ...

    # crud end