import os
from neo4j import GraphDatabase, Driver
from Neo4jDBMS.datatype import *
from typing import *
from Neo4jDBMS.statement import Statement

current_dir = os.path.dirname(__file__)
# ../..
db_dir = ''.join([ folder + os.sep for folder in current_dir.split(os.sep)[:-2] ])

DEFAULT_INFO_PATH = os.path.join(db_dir, 'db.json')

class DataBaseManager():
    r"""
        For neo4j db connection and base ctrl
        you can acess crud by create, search, update, delete method
        if you want to call basic query, use self.driver
        use close function to close connection
    """
    def __init__(self, info : Union[dict, str] = DEFAULT_INFO_PATH, data_base: str = "neo4j") -> None:
        r"""
            info dict is like:
            {
                "uri": "neo4j://localhost",
                "auth": ["username", "password"]
            }
            same as the db.json content
            you can input the db.json's path by info, which is default by DEFAULT_INFO_PATH
        """
        if type(info) is str:
            with open(info, 'r', encoding='UTF-8') as fd:
                db_info = json.load(fd)
            self.driver : Driver = GraphDatabase.driver(db_info["uri"], auth=tuple(db_info["auth"]))
        elif type(info) is dict:
            self.driver : Driver = GraphDatabase.driver(info["uri"], auth=tuple(info["auth"]))
        else:
            raise TypeError(f"Unkown info type: { type(info) }")
        self.driver.verify_connectivity()
        self.data_base = data_base

    def execute_query(self, query: str):
        records, summary, keys = self.driver.execute_query(
            query,
            age=42,
            database_=self.data_base,
        )
        return records, summary, keys
    
    def close(self) -> None:
        r""" close connection to database """
        self.driver.close()


    # crud begin @ToDo: compelete CRUD
    def create(self, item : Union[Union[Node, Edge], list[Union[Node, Edge]]]) -> None:
        r""" return a simple create a(n) node/entity or edge/relation statement """
        if type(item) is Iterable: # 添加一系列结点或边
            query_str = 'CREATE '.join([ str(i) + ',' for i in item ])
            self.execute_query(query_str) # 去掉最后的','
        elif type(item) is Node or type(item) is Edge: # 添加一个边
            self.execute_query(query_str)
        else: # 显然我们不能创造除结点和边之外的东西
            raise TypeError("You can only create node/entity or edge/relation")

    def createEntity(self, entity: Node) -> None:
        query = f'CREATE { entity }'
        self.execute_query(query)

    def search(self) -> Statement:
        r""" retrieve """
        ...
    
    def delete(self) -> Statement:
        ...
    
    def update(self) -> Statement:
        ...
    
    

    # crud end