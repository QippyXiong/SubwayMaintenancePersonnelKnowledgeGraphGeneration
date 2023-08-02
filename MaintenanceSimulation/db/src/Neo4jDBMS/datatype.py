"""
    包含了一些neo4j常用类型
"""

import json
from typing import Union

class Properties(dict):
    r"""
        properties of Node/Entity or Edge/Relation
    """
    def __init__(self, properties: dict = {}) -> None:
        super(Properties, self).__init__(properties)

    def __str__(self) -> str:
        if not any(self):
            return ''
        string = ''.join([ f'{key}:"{ self[key] }", ' for key in self])
        string = ' { ' + string[:-2] + ' }'
        return string

class Class:

    def __init__(self, class_name: str = None) -> None:
        self.class_name = class_name
    
    def __str__(self) -> str:
        if self.class_name:
            return ':' + self.class_name
        else:
            return '' 


class Variable:
    r""" 
        在一条Statement中，我们可以通过Variable来指定一个结点或边
        但Variable的内容不会实际存储到数据库中去
    """
    def __init__(self, variable_name: str = None) -> None:
        self.variable_name = variable_name

    def __str__(self) -> str:
        return self.variable_name if self.variable_name else ''


class Node:
    r"""
        type of Neo4j node, same as Entity
        variable has to be unique
    """
    def __init__(self, class_ : Union[Class, str] = Class(), properties : Properties = Properties(), variable: Union[str, Variable] = Variable()) -> None:
        r"""
            Class: 结点的类名
            properties: 结点属性
            variable: 在一条语句中，结点用何代替，只有在一条语句中反复使用该结点时使用
        """
        if type(class_) is str:
            class_ = Class(class_)
        self.class_ = class_
        # preventing parsing error
        if type(properties) is not Properties:
            raise TypeError("properties'type must be Properties")
        self.props = properties
        if type(variable) is str:
            variable = Variable(variable)
        self.variable = variable
        
    def __str__(self) -> str:
        
        return f'({ self.variable }{ self.class_ }{self.props})'
    

class Edge:
    r"""
        type of Neo4j edge, same as Relation
        variable has to be unique
    """
    def __init__(self, subject: Union[Node, Variable] = Node(), Class : Class = Class(), properties : Properties = Properties() , object: Union[Node, Variable] = Node(), variable: Variable = Variable()) -> None:
        """
            subject --edge-> object
        """
        if subject is Variable:
            subject = Node(subject)

        if object is Variable:
            object = Node(object)

        self.subject = subject
        self.object = object
        self.Class = Class
        # preventing parsing error
        if type(properties) == dict:
            properties = Properties(properties)
        elif type(properties) != Properties:
            raise TypeError("properties'type must be Properties")
        self.props = properties
        self.variable = variable


    def __str__(self) -> str:
        return f'{self.subject}-[{self.variable}:{self.Class}{self.props}]->{self.object}'


Entity = Node


Relation = Edge
