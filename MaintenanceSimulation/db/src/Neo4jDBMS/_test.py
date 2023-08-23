r"""
    test code for Neo4jDBMS package
"""

from os import path, sep as path_sep
from .__init__ import DataBaseManager
import json

current_dir = path.dirname(__file__)
# ../..
db_dir = ''.join([ folder + path_sep for folder in current_dir.split(path_sep)[:-2] ])

user_info = {}
with open(path.join(db_dir, 'db.json'), 'r', encoding='UTF-8') as fd:
    user_info = json.load(fd)

if __name__ == '__main__':
    dbms = DataBaseManager(user_info)
    """
        records, summary, keys = driver.execute_query(
            "MATCH (p:Person WHERE age = $age) RETURN p.name AS name",
            age=42,
            database_="neo4j",
        )
    """
    records, summary, keys = dbms.driver.execute_query(f"MATCH (p:Resource) RETURN p.uri as { 'uri' }", database_="neo4j")
    for item in records:
        for key in keys:
            print(f'{key}:{ item[key] }')

    print('===============================================')
    
    uri_head = "http://www.semanticweb.org/qippy/ontologies/2023/6/untitled-ontology-7/"
    records, summary, keys = dbms.driver.execute_query(f"match(n:Resource) where n.uri=~\"{ uri_head }.*\"  DETACH DELETE n", database_="neo4j")
    for item in records:
        for key in keys:
            print(f'{key}:{ item[key] }')

    print('==============================================')
    
    records, summary, keys = dbms.driver.execute_query(f"MATCH (p:Resource) RETURN p.uri as { 'uri' }", database_="neo4j")
    for item in records:
        for key in keys:
            print(f'{key}:{ item[key] }')

    dbms.driver.close()