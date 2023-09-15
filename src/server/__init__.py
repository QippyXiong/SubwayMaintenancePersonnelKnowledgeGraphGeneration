from typing import Union, Iterable

from fastapi import FastAPI
from pydantic import BaseModel
import json5

from neomodel import DateTimeFormatProperty, db, Relationship

# def main():
app = FastAPI()


from database import MaintenanceWorker, Capacity
from database.utils import EntityQueryByAtt, RelQueryByEnt, getRelEnt

@app.get("/")
def read_root():
    return "Hello"


@app.get("/items/{item_id}")
def read_item(item_id: str, q: Union[str, None] = None):
    print('fuck you')
    return {"item_id": item_id, "q": q}



class SearchData(BaseModel):
    properties  : dict
    relation    : str
@app.post("/search/entity/{ent_type}")
def read_entity(ent_type: str, data: SearchData):
    # return {"None":None, "[]":[]}
    # return {'ok': True, 'msg': 'success', 'data': ent_type}
    # print(ent_type)
    if len(data.properties) == 0:
        return {'ok': False, 'msg': 'properties is null', 'data': None}
    try:
        if data.relation == "None":
            ret_arr = EntityQueryByAtt(ent_type=ent_type, attr=data.properties)
            if isinstance(ret_arr, str):
                return {'ok': False, 'msg': ret_arr, 'data': None}
            if any(ret_arr):
                return {'ok': True, 'msg': 'success', 'data': ret_arr}
            else:
                return {'ok': False, 'msg': f'{ent_type} not exsists', 'data': None}
        if data.relation == "All":
            ret_arr = RelQueryByEnt(ent_type=ent_type, attr=data.properties, rel_type=None)
        else:
            ret_arr = RelQueryByEnt(ent_type=ent_type, attr=data.properties, rel_type=data.relation)
        if isinstance(ret_arr, str):
            return {'ok': False, 'msg': ret_arr, 'data': None}
        if any(ret_arr):
            return {'ok': True, 'msg': 'success', 'data': ret_arr}
        else:
            return {'ok': False, 'msg': data.relation + " not exsists" if data.relation else f"this {ent_type} has no relation", 'data': None}
    except Exception as e:
        return {'ok': False, "msg": str(e), 'data': None}

class MaintenaceWorkerSearchData(BaseModel):
    key: str
    data: str
@app.post("/search/maintenance_worker/person")
def read_worker(data: MaintenaceWorkerSearchData):
    key = data.key
    data = data.data
    # @TODO: 编写错误处理代码
    check = {key: data}
    ret_arr =[]
    try:
        # 人员属性字段查询
        persons = MaintenanceWorker.nodes.filter(**check)
        # 处理时间字段
        for person in persons:
            person_dict = dict()
            for key, _ in person.__all_properties__:
                person_dict[key] = str(getattr(person, key))
            ret_arr.append({"type": type(person).__name__, "record": person_dict})

        # 查询关联信息
        cap_name = []
        rec_infos = []
        for person in persons:
            capacities = person.capacity_rate.all()
            for cap in capacities:
                if cap.name not in cap_name:
                    cap_dict = dict()
                    for key, _ in cap.__all_properties__:
                        cap_dict[key] = str(getattr(cap, key))
                    ret_arr.append({"type": "Capacity", "record": cap_dict})
                    cap_name.append(cap.name)
                    # print(ret)
                query = f"""MATCH (p:MaintenanceWorker{{id : '{person.id}'}})<-[r:RATE] \
                     -(c:Capacity{{name : '{cap.name}'}}) RETURN r"""
                # for rel_name, rel in person.__all_relationships__:
                #     rel: Relationship
                #     related_nodes = rel.manager.all()
                #     for node in related_nodes:
                #         edge = rel.manager.relationship(node)
                r, _ = db.cypher_query(query)
                r = r[0][0]
                source = {"type":type(person).__name__, "majorkey":{"id":person.id}}
                target = {"type":type(cap).__name__, "majorkey":{"name":cap.name}}
                # rel = {"type": r.type, "record": {"source": person.id, "target": cap.name, "properties": r._properties}}
                rel = {"type": r.type, "record": {"source": source, "target": target, "properties": r._properties}}

                ret_arr.append(rel)

            maintenancerecords = person.maintenance_perform.all()
            for rec in maintenancerecords:
                # print("rec", rec)
                rec_info = {"malfunction": rec.malfunction, "place": rec.place, "malfunc_time": rec.malfunc_time}
                if rec_info not in rec_infos:
                    rec_dict = dict()
                    for key, _ in rec.__all_properties__:
                        rec_dict[key] = str(getattr(rec, key))
                    print(rec_dict)
                    ret_arr.append({"type": "MaintenanceRecord", "record": rec_dict})
                    rec_infos.append(rec_info)

                query = f"""MATCH (p:MaintenanceWorker{{id: '{person.id}'}})<-[r:PERFORMED]- \
                        (re:MaintenanceRecord{{malfunction: '{rec_info["malfunction"]}',          \
                        place: '{rec_info["place"]}',malfunc_time:'{rec_info["malfunc_time"]}'}}) \
                        RETURN r"""
                r, _ = db.cypher_query(query)
                r = r[0][0]
                source = {"type": type(person).__name__, "majorkey": {"id": person.id}}
                for key in rec_info.keys():
                    rec_info[key] = str(rec_info[key])
                target = {"type": type(rec).__name__, "majorkey": rec_info}
                rel = {"type": r.type,
                       "record": {"source": source, "target": target, "properties": r._properties}}
                ret_arr.append(rel)
        if any(ret_arr):
            return {'ok': True, 'msg': 'success', 'data': ret_arr}
        else:
            return {'ok': False, 'msg': 'person not exsists', 'data': None}

    except ValueError:
            return{'ok': False, 'msg': 'query key error', 'data': None}

@app.post("/search/maintenance_worker/capacity")
def read_worker(data: MaintenaceWorkerSearchData):
    key = data.key
    data = data.data
    # @TODO: 编写错误处理代码
    check = {key: data}
    ret_arr = []
    try:
        if key == "capacity":
            try:
                cap = Capacity.nodes.get(name=data)
            except Exception as e:
                return {'ok': False, 'msg': 'capacity not exsists', 'data': None}
            cap_dict = dict()
            for key, _ in cap.__all_properties__:
                cap_dict[key] = str(getattr(cap, key))
            ret_arr.append({"type": "Capacity", "record": cap_dict})

            persons = cap.rate.all()
            # 处理时间字段
            for person in persons:
                person_dict = dict()
                for key, _ in person.__all_properties__:
                    person_dict[key] = str(getattr(person, key))
                ret_arr.append({"type": type(person).__name__, "record": person_dict})
            return {'ok': True, 'msg': 'success', 'data': ret_arr}


    except ValueError:
            return{'ok': False, 'msg': 'query key error', 'data': None}

@app.get('/search/relations/{ent_type}')
def read_relations(ent_type: str):
    try:
        ret_val = getRelEnt(ent_type)
    except Exception as e:
        return { 'ok': False, 'msg': str(e), 'data': None }
    
    return {'ok': True, 'msg': 'success', 'data': ret_val}