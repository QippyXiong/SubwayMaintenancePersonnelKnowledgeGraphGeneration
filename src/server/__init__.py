from typing import Union, Iterable

from fastapi import FastAPI
from pydantic import BaseModel
import json5

from neomodel import DateTimeFormatProperty, db, Relationship

from dataclasses import dataclass

# def main():
app = FastAPI()


from database import MaintenanceWorker, Capacity
from database.utils import EntityQueryByAtt, RelQueryByEnt, getRelEnt, CreateEnt, kg_mapping, kg_majorkey_mapping, \
    handle_time_key, CreateRel, UpdateEnt, UpdateRel, DeleteEnt, DeleteRel


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

class CreateData(BaseModel):

    @dataclass
    class Relation:
        link_node_type: str
        link_node_properties: dict
        edge_type: str
        edge_properties: dict

    node_properties  : dict
    relation    : Union[None,list[Relation]]

@app.post("/create/entity&rel/{ent_type}")
def create_entity_rel(ent_type: str, data: CreateData):
    _, msg = CreateEnt(class_name=ent_type,attr=data.node_properties)
    if _ is None:
        return {'ok': False, 'msg': msg, 'data': None}
    else:
        if data.relation == "None":
            return {'ok': True, 'msg': msg, 'data': None}
        elif isinstance(data.relation, list):
            link_node_type = data.relation.link_node_type
            link_node_properties = data.relation.link_node_properties
            edge_type = data.relation.edge_type
            edge_properties = data.relation.edge_properties
            if link_node_type not in kg_mapping:
                msg = msg + "\n" + link_node_type + "does not exist"
                return {'ok': False, 'msg': msg, 'data': None}
            if edge_type not in kg_mapping:
                msg = msg + "\n" + edge_type + "does not exist"
                return {'ok': False, 'msg': msg, 'data': None}
            attr = handle_time_key(link_node_type, link_node_properties)
            link_entities = kg_mapping[link_node_type].nodes.filter(**attr)
            flag = True
            for e in link_entities:
                f, m = CreateRel(_, e, edge_type, edge_properties)
                msg = msg + m
                flag = flag & f
            return {'ok': flag, 'msg': msg, 'data': None}

class Rel_Data(BaseModel):
    start_node_type: str
    end_node_type: str
    start_node_properties: dict
    end_node_properties: dict
    edge_properties: dict

@app.post("/create/relationship/{rel_type}")
def create_rel(rel_type: str, data: Rel_Data):
    if data.start_node_type not in kg_mapping:
        msg = data.start_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if data.end_node_type not in kg_mapping:
        msg = data.end_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if rel_type not in kg_mapping:
        msg = rel_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    start_attr = handle_time_key(data.start_node_type, data.start_node_properties)
    end_attr = handle_time_key(data.end_node_type, data.end_node_properties)
    start_entities = kg_mapping[data.start_node_type].nodes.filter(**start_attr)
    end_entities = kg_mapping[data.end_node_type].nodes.filter(**end_attr)
    flag = True
    msg = ""
    for s in start_entities:
        for e in end_entities:
            f, m = CreateRel(s, e, rel_type, data.edge_properties)
            msg = msg + m
            flag = flag & f
    return {'ok': flag, 'msg': msg, 'data': None}

class Update_Ent_Data(BaseModel):
    properties     : dict
    new_properties : dict
@app.post("/update/entity/{ent_type}")
def update_entity(ent_type: str, data: Update_Ent_Data):
    _, msg = UpdateEnt(ent_type, data.properties, data.new_properties)
    if _ is None:
        return {'ok': False, 'msg': msg, 'data': None}
    else:
        return {'ok': True, 'msg': msg, 'data': None}

@app.post("/update/relation/{rel_type}")
def update_rel(rel_type: str, data: Rel_Data):
    if data.start_node_type not in kg_mapping:
        msg = data.start_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if data.end_node_type not in kg_mapping:
        msg = data.end_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if rel_type not in kg_mapping:
        msg = rel_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    start_attr = handle_time_key(data.start_node_type, data.start_node_properties)
    end_attr = handle_time_key(data.end_node_type, data.end_node_properties)
    start_entities = kg_mapping[data.start_node_type].nodes.filter(**start_attr)
    end_entities = kg_mapping[data.end_node_type].nodes.filter(**end_attr)
    flag = True
    msg = ""
    for s in start_entities:
        for e in end_entities:
            f, m = UpdateRel(start_entities, end_entities, rel_type, data.edge_properties)
            msg = msg + m
            flag = flag & f
    return {'ok': flag, 'msg': msg, 'data': None}

@app.post("/delete/relation/{rel_type}")
def delete_entity(rel_type: str, data: Rel_Data):
    if data.start_node_type not in kg_mapping:
        msg = data.start_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if data.end_node_type not in kg_mapping:
        msg = data.end_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if rel_type not in kg_mapping:
        msg = rel_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    start_attr = handle_time_key(data.start_node_type, data.start_node_properties)
    end_attr = handle_time_key(data.end_node_type, data.end_node_properties)
    start_entities = kg_mapping[data.start_node_type].nodes.filter(**start_attr)
    end_entities = kg_mapping[data.end_node_type].nodes.filter(**end_attr)
    flag = True
    msg = ""
    for s in start_entities:
        for e in end_entities:
            f, m = DeleteRel(start_entities, end_entities, rel_type)
            msg = msg + m
            flag = flag & f
    return {'ok': flag, 'msg': msg, 'data': None}

@app.post("/delete/entity/{ent_type}")
def delete_rel(ent_type: str, properties: dict):
    _, msg = DeleteEnt(ent_type, properties)
    if _:
        return {'ok': True, 'msg': msg, 'data': None}
    else:
        return {'ok': False, 'msg': msg, 'data': None}


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

from fastapi import WebSocket
from fastapi.responses import HTMLResponse

@app.get("/control")
async def sendShowData(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        await websocket.send_json()



r"""
below call LLM api
"""
from llm import gpt_maintainance_record_extraction
from database.utils import GenerateCapByRecord

class ExtractData(BaseModel):
    record: str


@app.post("/llm/extract/")
def extract_maintainance_record(data: ExtractData):
    r"""
    webserver post api for calling LLM for extracting maintainance record

    Args:
        data (ExtractData): only include record text for extracting
    """
    record = data.record
    try:
        infos = gpt_maintainance_record_extraction(record)
        print(infos)
        for info in infos:
            ok, msg = GenerateCapByRecord(info)
            if not ok:
                return {'ok': False, 'msg': msg, 'data': infos }
    except ValueError as e:
        return {'ok': False, 'msg': str(e), 'data': infos }
    return {'ok': True, 'msg': 'success', 'data': infos }