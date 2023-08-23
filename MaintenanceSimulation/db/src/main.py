from neo4j import GraphDatabase
import json5 as json
def change_type(byte):    
    if isinstance(byte,bytes):
        return str(byte,encoding="utf-8")  
    return json.JSONEncoder.default(byte)

class TestTransObj:
    
    def __init__(self) -> None:
        self.name = 'test_one'
        self.cla = 'try'
    




