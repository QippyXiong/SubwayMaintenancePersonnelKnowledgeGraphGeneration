from Neo4jDBMS.datatype import *
from neo4j import Driver, EagerResult

class Statement():
    r"""
        Many times we may want to corporate CRUD statements together, that's why we need statement class
        
        using corporate(other: Statement) to corporate two statement together

        using .execute() to execute statement
    """
    def __init__(self, manager, statement: str) -> None:
        r""" manager who will execute this statement, and manager should be DataBaseManager Class """
        self.driver = manager
        self.content = statement


    # @ToDo: Compelete Statement corporate
    def corporate(self, ) -> None:
        r""" 
            corporate two statement together, witch is a quite complex task

            Notice: I may not compele this function, before I delete this description you shouldn't use it
            
            type of other_statement must be Statement
        """
        raise NotImplementedError("corporate need tobe compeleted in Statement")

    def execute(self) -> EagerResult:
        r""" execute this statement, return the result """
        # return self.driver.execute_query(self.content)
        ...
    

class CreateStatement(Statement):
    r""" 意思是这是创建结点、创建边的语句，不是说创建一条语句 """
    def __init__(self, driver: Driver, statement: str) -> None:
        super().__init__(driver, statement)


