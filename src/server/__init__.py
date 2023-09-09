from typing import Union, Iterable

from fastapi import FastAPI
from pydantic import BaseModel
import json5

from neomodel import DateTimeFormatProperty

# def main():
app = FastAPI()


from database import MaintenanceWorker, Capacity


@app.get("/")
def read_root():
    return "Hello"


@app.get("/items/{item_id}")
def read_item(item_id: str, q: Union[str, None] = None):
    print('fuck you')
    return {"item_id": item_id, "q": q}


class MaintenaceWorkerSearchData(BaseModel):
    key: str
    data: str


@app.post("/search/maintenance_worker")
def read_worker(data: MaintenaceWorkerSearchData):
    key = data.key
    data = data.data
    # @TODO: 编写错误处理代码
    try:
        if(key == "id"):
            persons = MaintenanceWorker.nodes.get(id=data)
        elif(key == "name"):
            persons = MaintenanceWorker.nodes.filter(name=data).all()
        elif(key == "work_post"):
            persons = MaintenanceWorker.nodes.filter(work_post=data).all()
        elif(key == "capacity"):
            capacities = Capacity.nodes.get(name=data)
            persons = capacities.rate.all()

    except Exception as e:
        return {'msg': 'person not exsists'}

    if not isinstance(persons, Iterable):
        persons = [persons]

    ret_arr = []
    for person in persons:
        person_dict = dict()
        for key, _ in person.__all_properties__:
            person_dict[key] = str(getattr(person, key))
        ret_arr.append({
            'type': type(person).__name__,
            'record': person_dict
        })
    # if any(persons):

    return_arr = []


    #     return {'msg': 'success', 'data': 'nima' }
    #
    # else:
    return { 'ok': True, 'msg': 'person not exsists', 'data': ret_arr}


