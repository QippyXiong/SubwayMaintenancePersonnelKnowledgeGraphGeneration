"""
    包含了一些neo4j常用类型
"""

import json

class Properties(dict):
    r"""
        properties of Node/Entity or Edge/Relation
    """
    def __init__(self, properties: dict = {}) -> None:
        super(Properties, self).__init__(properties)

    def __str__(self) -> str:
        return  ' ' + json.dumps(self) if any(self) else ''


class Node:
    r"""
        type of Neo4j node, same as Entity
        label has to be unique
    """
    def __init__(self, label : str = '', Class : str = None, properties : Properties = Properties({})) -> None:
        r"""
            
        """
        self.label = label
        self.Class = Class
        # preventing parsing error
        if(type(properties) != Properties):
            raise TypeError("properties'type must be Properties")
        self.props = properties
        
    def __str__(self) -> str:
        return f'({ self.label }{ ":" + self.Class if self.Class else ""}{self.props})'
    

class Edge:
    r"""
        type of Neo4j edge, same as Relation
        Class has to be unique
    """
    def __init__(self, subject_label: str, Class : str, properties : Properties , object_label: str) -> None:
        self.subject = Node(subject_label)
        self.object = Node(object_label)
        self.Class = Class
        # preventing parsing error
        if type(properties) == dict:
            properties = Properties(properties)
        elif type(properties) != Properties:
            raise TypeError("properties'type must be Properties")
        self.props = properties


    def __str__(self) -> str:
        return f'{self.subject}-[:{self.Class}{self.props}]->{self.object}'


Entity = Node


Relation = Edge
