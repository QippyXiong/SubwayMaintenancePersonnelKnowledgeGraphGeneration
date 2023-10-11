r"""
This file includes code for extracting information based on prompt-engineering

at now we using openai chatgpt to extract information from text
"""
from .utils import extract_json


from .gpt import chat, chat_stream

def gpt_maintainance_record_prompt(content: str)-> str:
	r"""
	Args:
		content: maintaince text for extracting information

	Returns:
		prompt for openai chatgpt
	"""
	GPT_maintainace_prompt = r"""
你的任务是从地铁人员维修记录本文中提取出这个记录中包含的人物、他们的岗位、他们完成的维修内容、此次维修所在的地点、维修开始时间、维修结束时间、维修持续时间。

你需要将你的输出结果组织成一个json数组，数组中的一个元素的格式如下：
{
	"person": 维修人员名称,
	"station": 人员的岗位,
	"malfunc":维修故障,
	"content": 维修内容,
	"place": 维修所在地点,
	"begin_time": 维修开始时间,
	"end_time": 维修结束时间,
	"duration": 维修持续时间
}
你应当将文本中未提及的字段标记为null
你应该将维修不同的设备视为不同的维修内容，同时将每位维修人员的每一次维修任务的维修内容分别记录在不同的json数组的content中，而不是简单的填充在同一个json数组的content中
维修故障应当是以下数组中的元素之一，维修故障数组格式如下：{轨道损坏,轮胎车轴故障,车门故障,照明损坏,空调故障,制动系统故障,排水沟损坏,排水系统堵塞,通风系统堵塞,烟雾报警器故障,紧急停车系统故障,自动售票机故障,安检设备故障,闸机故障,电梯故障,扶梯故障,电力系统故障,地铁信号故障,监视系统故障}
对于不在维修故障数组中的故障，标类为'其它'
参与辅助工作的人员维修内容填写"辅助工作"，其它维修事件记录内容(维修故障，维修开始时间，维修结束时间，维修持续时间等)与主要维修人员的维修事件记录内容保持一致
人员岗位文本中未提及则填充null,维修人员的人员岗位应该为以下数组的元素之一：{轨道维修工程师,车辆维修技术员,排水与通风技术员,安全检测技术员,设备维护技术员,电力维修技术员,通信维护员}
时间格式是：%Y-%m-%d %H:%M:%S
示例文本如下：2021.5.21,早八点龙凤溪地铁站空调故障，车辆维修技术员XXX前往维修，于早十点维修完成，张三参与了辅助工作。
输出内容如下：
[
	{
		"person": "张三",
		"station": "车辆维修技术员",
		"malfunc": "空调故障",
		"content": "空调维修",
		"place": "小寨站",
		"begin_time": "2021-05-21 08:00:00",
		"end_time": "2021-05-21 10:00:00",
		"duration": "2小时"
	},
	{
		"person": "张三",
		"station": null,
		"malfunc": "空调故障",
		"content": "辅助工作",
		"place": "小寨站",
		"begin_time": "2021-05-21 08:00:00",
		"end_time": "2021-05-21 10:00:00",
		"duration": "2小时"
	},
]
你需要处理的文本：
"""
	return GPT_maintainace_prompt + content

import json5


def gpt_maintainance_record_extraction(content: str) -> dict:
	r"""
	Args:
		content: maintaince text for extracting information

	Returns:
		dict of information extracted from text

	Throws:
		ValueError: if extraction failed, return msg contains gpt return value
	"""
	prompt = gpt_maintainance_record_prompt(content)
	res_text = chat(prompt, num_choices=1, role='user')
	res = extract_json(res_text[0])
	if not res:
		raise ValueError('extraction failed, return text is {}'.format(res_text[0]))
	return res