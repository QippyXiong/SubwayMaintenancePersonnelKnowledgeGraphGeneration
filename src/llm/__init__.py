
from pathlib import Path

import gpt

from database.utils import GenerateCap

api_key_path = Path(__file__).parent.parent.parent.joinpath('config', 'openai', 'apikey.json')

def connect_llm(llm_name: str, **kwargs):
	r"""
	连接到llm，执行完此方法后应当能够执行任意的函数
	llm_name输入后，需要指定连接信息，用kwargs来指定，缺少会抛出KeyError
	目前只支持openai

	Args:
		llm_name: name of LLM you want to use

	list of LLM and need params:
		openai: needs api_key(openai chatgpt api key), api_base(url for access openai model)

	"""
	if llm_name == 'openai':
		gpt.connect_openai(api_key=kwargs['api_key'], api_base=kwargs['api_base'])
	else:
		raise ValueError(f"unkown LLM name { llm_name }")
	
	

import json

def main():
	content = r"""
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

2023年7月28日，上午八点发现小寨站处铁轨损发生破损，同时造成了地铁车辆车轮损坏，铁轨维修人员蔡海波立刻前往进行维修，维修持续了三个小时，修复了铁轨的破损情况，使得地铁能够继续正常运行；同时车辆维修人员平如愿前往查看车辆情况，不仅维修了车轮，而且修复了地铁车厢的车窗问题，在下午两点钟完成了车辆的全部维修工作，两次任务中，维修人员熊深豪参与了辅助工作。
"""
	with open(api_key_path) as fp:
		api_status = json.load(fp)
		print(api_status)
		gpt.connect_openai(**api_status)

	for r in gpt.chat(content):
		print(r)
		data = json.loads(r)

	# print(data)

	return data

if __name__ == '__main__':
	# r = main()
	r = {
		'person': '蔡海波',
		'station': '铁轨维修人员',
		'malfunc': '轨道损坏',
		'content': '轨道维修',
		'place': '小寨站',
	 	'begin_time': '2023-07-28 08:00:00',
		'end_time': '2023-07-28 11:00:00',
		'duration': '3小时'
	}
	_, msg = GenerateCap(r)
	print('main return:', msg)