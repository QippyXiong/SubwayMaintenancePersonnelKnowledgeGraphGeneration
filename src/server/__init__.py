from fastapi import FastAPI, Request, Depends
from typing import Union
from pydantic import BaseModel
from neomodel import db

# def main():
app = FastAPI()

@app.get("/")
def read_root():
    return "Hello"


@app.get("/items/{item_id}")
def read_item(item_id: str, q: Union[str, None] = None):
    print('fuck you')
    return {"item_id": item_id, "q": q}


class MaintenaceWorkerSearchData(BaseModel):
    id: str
    

from database import MaintenanceWorker

@app.post("/search/maintenance_worker")
def read_worker(data: MaintenaceWorkerSearchData):
    # @TODO: 编写错误处理代码
    try:
        persons = MaintenanceWorker.nodes.filter(id=data.id)
    except Exception as e:
        print(e.with_traceback())
        return { 'msg': 'person not exsists' }

    if any(persons): 
        return { 'msg': 'success', 'data': persons[0] }
    else:
        return { 'msg': 'person not exsists', 'data': None }

