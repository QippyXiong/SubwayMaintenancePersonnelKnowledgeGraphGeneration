r"""
一些工具方法，目前包含：
1.读人工制定excel文件内容到图谱中

"""
import datetime
from typing import Union, Dict, List, Optional

import pandas as pd
from neo4j.exceptions import ServiceUnavailable
from neomodel import db, Relationship, StructuredNode, RelationshipManager, StructuredRel
from neomodel.exceptions import DeflateError
from neomodel.match import NodeSet

from .graph_models.maintenance_personnel import MaintenanceWorker, MaintenanceRecord, Capacity, CapacityRate, \
	MaintenancePerformance

kg_mapping = {
	"MaintenanceWorker"		: MaintenanceWorker,
	"Capacity"		   		: Capacity,
	"MaintenanceRecord"		: MaintenanceRecord,
	"CapacityRate"			: CapacityRate,
	"MaintenancePerformance": MaintenancePerformance
}
kg_majorkey_mapping = {
	"MaintenanceWorker"		: ["id"],
	"Capacity"		   		: ["name"],
	"MaintenanceRecord"		: ["malfunction", "place", "malfunc_time"]
}

def GetEntAttribute(class_name:str):
	r"""
	Args:
		'class_name':str   # 类名
	Returns:
		'ret':[[属性名，字段类型]]
	"""
	ent_class = kg_mapping[class_name]
	atts = ent_class.__all_properties__
	ret = list()
	for att, ty in atts:
		ret.append([att,type(ty).__name__])
	return ret

def CreateEnt(class_name:str,attr:dict):
	r"""根据类名和属性值创建实体"""
	try:
		ent_class = kg_mapping[class_name]
		major_key = {mk: attr[mk] for mk in kg_majorkey_mapping[class_name]}
		major_key = handle_time_key(class_name, major_key)
	except KeyError as e:
		print(str(e), "not exist")
		return
	only_ent = ent_class.nodes.filter(**major_key)
	if(only_ent.__nonzero__()):
		msg = class_name + str(major_key) +" is already exist"
		return msg
	else:
		attr = handle_time_key(class_name, attr)
		new_ent = ent_class(**attr)
		new_ent.save()
		msg = "create " + class_name + str(major_key) + " succeed"
		return msg

def CreateRel(start_ent: StructuredNode, end_ent: StructuredNode, rel_name:str, attr:dict):
	rel: RelationshipManager = getattr(start_ent, "CapacityRate")
	edge = rel.connect(end_ent, attr)

def DeleteEnt(class_name:str,attr:dict):
	try:
		ent_class = kg_mapping[class_name]
	except KeyError as e:
		msg = str(e) +  "not exist"
		return msg
	attr = handle_time_key(class_name, attr)
	del_ent = ent_class.nodes.filter(**attr)
	if(del_ent.__nonzero__()):
		nums = del_ent.__len__()
		for e in del_ent:
			e.delete()
			# s = class_name+str(parse_record_to_dict(e))
			# msg.append(s)
		return str(nums) +" "+ class_name + str(attr) + " is already deleted"
	else:
		msg = class_name + str(attr) + " not exist"
		return msg

def UpdateEnt(class_name:str, attr:dict, new_attr:dict):
	r"""
		eg: 匹配 attr:{id:"3456",name:"张三"}，修改 new_attr{name:"李四"}
		修改主键： 判断修改后是否重复
	"""

	try:
		ent_class = kg_mapping[class_name]
	except KeyError as e:
		msg = str(e)+ "not exist"
		return msg

	try:
		# 修改主键
		# ##修改后的实体已存在
		major_key = {mk: new_attr[mk] for mk in kg_majorkey_mapping[class_name]}
		major_key = handle_time_key(class_name, major_key)
		only_ent = ent_class.nodes.filter(**major_key)
		if (only_ent.__nonzero__()):
			msg = class_name + str(major_key) + " is already exist"
			return msg
		else:
			# 进行修改
			attr = handle_time_key(class_name, attr)
			update_ent = ent_class.nodes.filter(**attr)
			nums = update_ent.__len__()
			if nums == 1:
				# 修改单节点
				# ##更新属性值
				update_ent_ = update_ent[0]
				new_attr = handle_time_key(class_name, new_attr)
				for key, value in new_attr.items():
					setattr(update_ent_, key, value)
				# print(parse_record_to_dict(update_ent_))
				update_ent_.save()
				msg = "The primary key of single entity has been modified"
				return msg
			elif nums > 1:
				# 不能同时修改多个节点的主键
				msg = "The primary key of multiple entities cannot be modified simultaneously"
				return msg
			else:
				# 需要修改的节点不存在
				msg = "The single entity that needs to be modified, "+class_name + str(attr) + ", does not exist"
				return msg
	except KeyError as e:
		# 修改非主键
		attr = handle_time_key(class_name, attr)
		new_attr = handle_time_key(class_name, new_attr)
		update_ent = ent_class.nodes.filter(**attr)
		nums = update_ent.__len__()
		if nums > 0:
			for e in update_ent:
				# ##更新属性值
				for key, value in new_attr.items():
					setattr(e, key, value)
				e.save()
		return str(nums) +" "+ class_name + str(attr) + " is already updated"

def UpdateRel(start_ent: StructuredNode, end_ent: StructuredNode, rel_name:str, attr:dict):
	rel: RelationshipManager = getattr(start_ent, rel_name)
	edge = rel.relationship(end_ent)
	for key, value in attr.items():
		setattr(edge, key, value)
	# print(parse_record_to_dict(edge))
	edge.save()


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
			rel = record.MaintenancePerformance.connect( MaintenanceWorker.nodes.get(id=row_dict['工号']), {
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
			worker2capacity = capacity.CapacityRate.connect(MaintenanceWorker.nodes.get(id=row_dict['维保人员工号']), {
				'level': row_dict['维修能力等级'],
			 } )
			worker2capacity.save()
		except DeflateError:
			pass

def parse_record_to_dict(record: Union[Relationship, StructuredNode, StructuredRel]) -> Dict:
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


def getRelEnt(class_name: str):
	r"""
	Args:
		'class_name':str   # 类名
	Returns:
		'ret':[[]]
		[关系名，尾实体类名]
	"""
	ret = []
	start_ent_class = kg_mapping[class_name]
	# print(MaintenanceWorker.__all_relationships__)
	for rel_name, _ in start_ent_class.__all_relationships__:
		rel: RelationshipManager = getattr(start_ent_class, rel_name)
		ret.append([rel_name, rel._raw_class])
	return ret

# def getRelNameAndEntName(ent:StructuredNode):
# 	ret = []
# 	for rel_name, _ in ent.__all_relationships__:
# 		rel: RelationshipManager = getattr(ent, rel_name)
# 		targetEnt = type(rel[0]).__name__
# 		ret.append([rel_name,targetEnt])
# 	return ret
def EntityQueryByAtt(ent_type:str, attr:dict):
	r"""
	通过实体属性查询实体并返回实体所有属性值
	"""
	ret_arr = []
	relations = getRelEnt(ent_type)
	try:
		# 时间类型字段处理
		# ent_class = kg_mapping[ent_type]
		# time_key = get_time_key(ent_class)
		# for k in attr.keys():
		# 	if k in time_key:
		# 		try:
		# 			attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d %H:%M:%S")
		# 		except ValueError:
		# 			msg = "time format error, it should be %Y-%m-%d %H:%M:%S"
		# 			return msg
		attr = handle_time_key(ent_type, attr)

		entities = kg_mapping[ent_type].nodes.filter(**attr)
		for ent in entities:
			ent_dict = parse_record_to_dict(ent)
			record = {"element_id": ent.element_id, "properties":ent_dict, "relations": relations}
			ret_arr.append({"type": type(ent).__name__, "record": record})
		return ret_arr
	# except DeflateError:
	# 	msg = f"{ent_type} not exist"
	# 	return msg
	except ValueError:
		msg = "property key error"
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
		# ent_class = kg_mapping[ent_type]
		# time_key = get_time_key(ent_class)
		# for k in attr.keys():
		# 	if  k in time_key:
		# 		attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d %H:%M:%S")
		attr = handle_time_key(ent_type, attr)

		entities = kg_mapping[ent_type].nodes.filter(**attr)
		for ent in entities:
			if rel_type is None:
				for rel_name, _ in ent.__all_relationships__:
					# print(rel_name)
					rel: RelationshipManager = getattr(ent, rel_name)
					ret_arr.extend(RelQueryByRel(rel_name, rel))
			else:
				try:
					rel: RelationshipManager = getattr(ent, rel_type)
					ret_arr.extend(RelQueryByRel(rel_type, rel))
				except AttributeError:
					# 关系类型错误
					msg = "relationship key error"
					# print(msg)
					return msg
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

		# print("test" ,type(node).__name__,node.element_id,parse_record_to_dict(node))

		record2 = {"element_id":node.element_id,"record":parse_record_to_dict(node)}
		ret_arr.append({"type": type(node).__name__, "record": record2})
		# if source['type'] == rel_type:
		# 	record2 = {"element_id": edge._end_node_element_id, "record": parse_record_to_dict(edge.end_node())}
		# 	ret_arr.append({"type": type(edge.end_node()).__name__, "record": record2})
		# else:
		# 	record2 = {"element_id": edge._start_node_element_id, "record": parse_record_to_dict(edge.start_node())}
		# 	ret_arr.append({"type": type(edge.start_node()).__name__, "record": record2})
	return ret_arr

def get_time_key(ent_class: Union[StructuredNode, StructuredRel]):
	r"""
	得到类的时间属性字段
	"""
	attributes = ent_class.__all_properties__
	time_att = []
	for att_name, att_value in attributes:
		if type(att_value).__name__ in ['DateProperty', 'DateTimeFormatProperty']:
			time_att.append((att_name, type(att_value).__name__))
	return time_att

def handle_time_key(ent_type: str, attr: Dict):
	r"""
	时间类型字段处理
	"""
	ent_class = kg_mapping[ent_type]
	time_key = get_time_key(ent_class)
	# for k in attr.keys():
	# 	if k in time_key:
	# 		attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d %H:%M:%S")
	attr_keys = list(attr.keys())
	for k, t in time_key:
		if k not in attr_keys: continue # @TODO: optimize
		if t == 'DateProperty':
			attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d")
		elif t == 'DateTimeFormatProperty':
			attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d %H:%M:%S")
	return attr