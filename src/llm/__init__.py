
from pathlib import Path

import gpt

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
	"content": 完成的维修内容,
	"place": 维修所在地点,
	"begin_time": 维修开始时间,
	"end_time": 维修结束时间,
	"durationi": 维修持续时间
}
你应当将文本中未提及的字段标记为null

你需要处理的文本：

2023年7月28日，上午八点发现小寨站处铁轨损发生破损，同时造成了地铁车辆车轮损坏，铁轨维修人员蔡海波立刻前往进行维修，维修持续了三个小时，修复了铁轨的破损情况，使得地铁能够继续正常运行；同时车辆维修人员平如愿前往查看车辆情况，不仅维修了车轮，而且修复了地铁车厢的车窗问题，在下午两点钟完成了车辆的全部维修工作，两次任务中，维修人员熊深豪参与了辅助工作。
"""
	with open(api_key_path) as fp:
		api_status = json.load(fp)
		print(api_status)
		gpt.connect_openai(**api_status)

	for r in gpt.chat(content):
		print(r)

	return

if __name__ == '__main__':
	r = main()
	print('main return:', r)