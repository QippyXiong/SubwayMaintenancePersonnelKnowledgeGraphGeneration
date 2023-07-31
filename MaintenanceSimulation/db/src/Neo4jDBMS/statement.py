from datatype import *
from neo4j import Driver, EagerResult

class Statement():
    r"""
        Many times we may want to corporate CRUD statements together, that's why we need statement class
        
        using corporate(other: Statement) to corporate two statement together

        using .execute() to execute statement
    """
    def __init__(self, driver: Driver, statement: str) -> None:
        r""" driver who will execute this statement """
        self.driver = driver
        self.content = statement


    # @ToDo: Compelete Statement corporate
    def corporate(self, other_statement) -> None:
        r""" 
            corporate two statement together, witch is a quite complex task

            Notice: I may not compele this function, before I delete this description you shouldn't use it
            
            type of other_statement must be Statement
        """
        if type(other_statement) is not Statement:
            raise TypeError("You can only corporate two statements together")
        ...

    
    def execute(self) -> EagerResult:
        r""" execute this statement, return the result """
        return self.driver.execute_query(self.content)
    
