r"""
一些工具方法，目前包含：
1.读人工制定excel文件内容到图谱中

"""
from typing import Union, Dict, List, Optional

import pandas as pd
from neo4j.exceptions import ServiceUnavailable
from neomodel import db, Relationship, StructuredNode, RelationshipManager, StructuredRel
from neomodel.exceptions import DeflateError

from .graph_models.maintenance_personnel import MaintenanceWorker, MaintenanceRecord, Capacity

kg_mapping = {
	"MaintenanceWorker"	: MaintenanceWorker,
	"Capacity"		   		: Capacity,
	"MaintenanceRecord"		: MaintenanceRecord
}

def load_excel_file_to_graph(file_path: str):
	try:
		db.cypher_query(
			r"""
			MATCH(n)
			DETACH DELETE n
			"""
		) # 删掉原先图谱中的全部内容
	except ServiceUnavailable:
		print("[Neomodel Error] 未能连接到neo4j服务器，请检查neo4j服务器是否开启")
		return

	mapping_worker = {
		'id' 				: '工号/志愿者编号',
		'name'				: '姓名',
		'sex' 				: '性别',
		'nation'			: '民族',
		'phone'				: '联系方式',
		'birth'				: '出生日期',
		'live_in'			: '居住地址',
		'employ_date' 		: '入职时间',
		'work_post' 		: '岗位',
		'work_level'		: '岗位级别',
		'department' 		: '部门',
	}
	# mapping_worker = { mapping_worker[key]: key for key in mapping_worker }

	mapping_record = {
		# 'id'				: '工号',
		'malfunction' 		: '故障类型',
		'place'				: '故障位置',
		'malfunc_time'		: '故障上报时间',
		'begin_time'		: '维修开始时间',
		'complish_time'		: '维修完成时间',
		'review'			: '定期检修记录',
	}

	mapping_capacity = {
		'name' 			: '维修能力名称',
		'description' 	: '描述',
		'rule'			: '规则',
	}

	# mapping_record = { mapping_record[key]: key for key in mapping_record }

	# 处理维保人员数据
	query = r"""CREATE CONSTRAINT MaintenanceWork_unique_key 
			FOR(m: MaintenanceWorker) REQUIRE(m.id) IS UNIQUE
	"""

	worker_data = pd.read_excel(file_path, sheet_name='维保人员')

	for row in worker_data.itertuples():
		data_dict = mapping_worker.copy()
		row_dict = { worker_data.keys()[i-1] : v for i, v in enumerate(row) }
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		try:
			worker = MaintenanceWorker.nodes.get(id = data_dict['id'])
		except Exception as e:
			worker = MaintenanceWorker(**data_dict)
			worker.phone = str(worker.phone)
			worker.save()

	# 处理维修记录数据
	records = pd.read_excel(file_path, sheet_name='维修记录')

	for row in records.itertuples():
		data_dict = mapping_record.copy()
		row_dict = {records.keys()[i-1] : v for i, v in enumerate(row)}
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		try:
			# 查询维修记录是否已存在
			record = MaintenanceRecord.nodes.get(
				malfunction = data_dict['malfunction'],
				place 		= data_dict['place'],
				malfunc_time= data_dict['malfunc_time'],
				)
			# print(record)

			# 查询维修记录是否未关联此条记录的维修人员
			record2worker = record.perform.all()
			ids = [w.id for w in record2worker]
			if row_dict['工号'] not in ids:
				rel = record.MaintenancePerformance.connect(MaintenanceWorker.nodes.get(id=row_dict['工号']), {
					'malfunc_type': record.malfunction,  # 维修记录故障内容记录故障类型
					'performance': record.review  # 维修记录返修评价记录维修效果
				})
				rel.save()
		except Exception as e:
			record = MaintenanceRecord(**data_dict)
			# print("ttt",record)
			record.save()
			rel  = record.MaintenancePerformance.connect( MaintenanceWorker.nodes.get(id=row_dict['工号']), {
				'malfunc_type': record.malfunction,  # 维修记录故障内容记录故障类型
				'performance': record.review  # 维修记录返修评价记录维修效果
			 } )
			rel.save()


	#处理维修能力数据
	Mcapacities = pd.read_excel(file_path, sheet_name='维修能力')

	for row in Mcapacities.itertuples():
		data_dict = mapping_capacity.copy()
		row_dict = { Mcapacities.keys()[i-1] : v for i, v in enumerate(row) }
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		try:
			capacity = Capacity.nodes.get(name=data_dict['name'])
		except Exception as p:
			capacity = Capacity(**data_dict)
			capacity.save()
		try:
			worker2capacity  = capacity.CapacityRate.connect(MaintenanceWorker.nodes.get(id=row_dict['维保人员工号']), {
				'level': row_dict['维修能力等级'],
			 } )
			worker2capacity.save()
		except DeflateError:
			pass

def parse_record_to_dict(record: Union[Relationship, StructuredNode]) -> Dict:
	r"""
	Args:
		'record': entity | relationship
	将特殊属性字段转换成字符串，如时间格式
	"""
	props: Dict = record.__properties__
	props.popitem()
	for k in props:
		props[k] = str(props[k])
	return props

# def EntityQueryByElement_id(ent_type:str, attr:dict)->List:
# 	r"""
# 	通过实体属性查询实体并返回实体所有属性值
# 	"""
# 	ret_arr = []
# 	try:
# 		entities = kg_mapping[ent_type].nodes.filter(**attr)
# 		for ent in entities:
# 			ent_dict = parse_record_to_dict(ent)
# 			ret_arr.append({"type": type(ent).__name__, "record": ent_dict})
# 		return ret_arr
# 	except DeflateError:
# 		return []
def EntityQueryByAtt(ent_type:str, attr:dict):
	r"""
	通过实体属性查询实体并返回实体所有属性值
	"""
	ret_arr = []
	try:
		entities = kg_mapping[ent_type].nodes.filter(**attr)
		for ent in entities:
			ent_dict = parse_record_to_dict(ent)
			ret_arr.append({"type": type(ent).__name__, "record": {"element_id": ent.element_id, "properties":ent_dict}})
		return ret_arr
	except DeflateError:
		msg = "property key error"
		return msg
	except ValueError:
		msg = f"{ent_type} not exist"
		return msg

def RelQueryByEnt(ent_type:str, attr:dict, rel_type:Optional[str]):
	r"""
	Args:	'ent_type':str,
			'attr'	:dict
	Returns:
			List
	输入实体查询，返回相关联的实体及其关系边
	"""
	ret_arr = []
	try:
		entities = kg_mapping[ent_type].nodes.filter(**attr)
		for ent in entities:
			if rel_type is None:
				for rel_name, _ in ent.__all_relationships__:
					print(rel_name)
					rel: RelationshipManager = getattr(ent, rel_name)
					ret_arr.extend(RelQueryByRel(rel_name, rel))
			else:
				try:
					rel: RelationshipManager = getattr(ent, rel_type)
					ret_arr.extend(RelQueryByRel(rel_type, rel))
				except AttributeError:
					# 关系类型错误
					msg = "key error"
					print(msg)
					return None
					# return msg
		return ret_arr
	except DeflateError:
		pass
def RelQueryByRel(rel_type: str, rel :RelationshipManager):
	ret_arr = []
	for node in rel.all():
		edge = rel.relationship(node)
		source = {"type": type(edge.start_node()).__name__, "element_id": edge._start_node_element_id}
		target = {"type": type(edge.end_node()).__name__, "element_id": edge._end_node_element_id}
		properties = parse_record_to_dict(edge)
		record1 = {"source": source, "target": target, "properties": properties}
		ret_arr.append({"type": type(edge).__name__, "record": record1})

		if source['type'] == rel_type:
			record2 = {"element_id": edge._end_node_element_id, "record": parse_record_to_dict(edge.end_node())}
			ret_arr.append({"type": type(edge.end_node()).__name__, "record": record2})
		else:
			record2 = {"element_id": edge._start_node_element_id, "record": parse_record_to_dict(edge.start_node())}
			ret_arr.append({"type": type(edge.start_node()).__name__, "record": record2})
	return ret_arr