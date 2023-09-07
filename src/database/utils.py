r"""
一些工具方法，目前包含：
1.读人工制定excel文件内容到图谱中

"""

import pandas as pd

from .graph_models.maintenance_personnel import MaintenanceWorker, MaintenanceRecord, MaintenancePerformance
from neomodel import db

def load_excel_file_to_graph(file_path: str):

	db.cypher_query(
		r"""
		MATCH(n)
		DETACH DELETE n
		"""
	) # 删掉原先图谱中的全部内容

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

	# mapping_record = { mapping_record[key]: key for key in mapping_record }

	# 处理维保人员数据
	worker_data = pd.read_excel(file_path, sheet_name='维保人员')

	for row in worker_data.itertuples():
		data_dict = mapping_worker.copy()
		row_dict = { worker_data.keys()[i-1] : v for i, v in enumerate(row) }
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		worker = MaintenanceWorker(**data_dict)
		worker.phone = str(worker.phone)
		worker.save()

	# 处理维修记录数据
	records = pd.read_excel(file_path, sheet_name='维修记录')

	for row in records.itertuples():
		data_dict = mapping_record.copy()
		row_dict = { records.keys()[i-1] : v for i, v in enumerate(row) }
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		# worker = MaintenanceWorker.nodes.get(id=data_dict['id'])
		record = MaintenanceRecord(**data_dict)
		record.save()
		rel  = record.perform.connect( MaintenanceWorker.nodes.get(id=row_dict['工号']), { 
			'malfunc_type': record.malfunction,
			'performance': record.review
		 } )
		rel.save()
		
		