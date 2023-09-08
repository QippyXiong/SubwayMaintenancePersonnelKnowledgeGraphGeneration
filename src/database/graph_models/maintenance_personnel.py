from neomodel import (
	config, 
	StructuredNode, StringProperty, StructuredRel, 
	UniqueIdProperty, RelationshipTo, RelationshipFrom,
	DateProperty, IntegerProperty, BooleanProperty, DateTimeFormatProperty,
	
	)


class MaintenanceWorker(StructuredNode):
	r"""
	维修人员实体
	"""
	id 				= StringProperty(index=True, max_length=20)			# 工号/志愿者编号
	name			= StringProperty(index = True, max_length=32)		# 姓名
	sex 			= BooleanProperty()									# 性别
	nation			= StringProperty(max_length=20)						# 民族
	phone			= StringProperty(max_length=11)						# 联系方式
	birth			= DateProperty()									# 出生日期
	live_in			= StringProperty(max_length=256)					# 居住地址
	employ_date 	= DateProperty()									# 入职时间
	work_post 		= StringProperty()									# 岗位
	work_level		= StringProperty()									# 岗位级别
	department 		= StringProperty(max_length=20)						# 部门


class MaintenancePerformance(StructuredRel):
	r"""
	维修记录表现边
	"""
	malfunc_type 	= StringProperty()	# 故障类型
	performance		= StringProperty()	# 维修效果


class MaintenanceRecord(StructuredNode):
	r"""
	维修记录实体
	"""
	malfunction 	= StringProperty()			# 故障内容
	place			= StringProperty()			# 故障位置
	malfunc_time	= DateTimeFormatProperty()	# 故障上报时间
	begin_time		= DateTimeFormatProperty()	# 维修开始时间
	complish_time	= DateTimeFormatProperty()	# 维修完成时间
	review			= StringProperty()			# 返修评价
	
	perform = RelationshipFrom('MaintenanceWorker', 'PERFORMED', model=MaintenancePerformance)


class SkillAssessResult(StructuredRel):
	r"""
	考核结果边
	"""
	result		= StringProperty()


class SkillAssessment(StructuredNode):
	r"""
	技能考核实体
	"""
	skill_type		= StringProperty()
	assess_date		= DateProperty()


class CapacityRate(StructuredRel):
	level		= StringProperty()


class Capacity(StructuredNode):
	id			= UniqueIdProperty()
	name 		= StringProperty(max_length=10)
	description = StringProperty(max_length=256)
	rule		= StringProperty()

	rate		= RelationshipFrom('MaintenanceWorker', 'RATE', model=CapacityRate)

