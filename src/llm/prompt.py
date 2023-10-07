from typing import Union
from dataclasses import dataclass


class Prompt:
	r"""
	Prompt for LLM

	
	"""
	


def extract_json(str: str, encoding='UTF-8') -> Union[dict, None]:
	r"""
	find the json string in str and parse it
	"""
	import json5
	r"""
	logic: using stack accept '{', '}' or '[', ']', parse the final and max '{-}' / '[-]' ones
	"""
	stack = []
	result = {}
	for i in range(len(str)):
		if str[i] == '{' or str[i] == '[':
			stack.append((str[i], i))
		
		if str[i] == '}':
			if not stack: # empty list, only happen if '}' appears before json string
				continue

			char, idx = stack.pop()
			
			if char == '{': # matched, parse
				try:
					result = json5.loads(str[idx: i+1])
					print('one: ', result)
				except:
					stack = [] # parse fail, this not json, that mean all before i cannot be json
			else:  # that mean it's '[', all these context are errored
				stack = []
		
		if str[i] == ']': # same as the previous one
			if not stack:
				continue

			char, idx = stack.pop()

			if char == '[':
				try:
					result = json5.loads(str[idx: i+1])
					print('one: ', result)
				except:
					stack = []
			else:
				stack = []
	return result

if __name__ == '__main__':

	print( extract_json(
	r"""
根据提供的文本，我将提取相关信息并组织成一个JSON数组：

```json
[
  {
    "person": "蔡海波",
    "station": "铁轨维修人员",
    "content": "修复了铁轨的破损情况",
    "place": "小寨站",
    "begin_time": "2023年7月28日 上午八点",
    "end_time": null,
    "duration": "3小时"
  },
  {
    "person": "平如愿",
    "station": "车辆维修人员",
    "content": "维修了车轮，修复了地铁车厢的车窗问题",
    "place": "未提及",
    "begin_time": "2023年7月28日 下午两点",
    "end_time": "未提及",
    "duration": "未提及"
  },
  {
    "person": "熊深豪",
    "station": "维修助手",
    "content": "辅助工作",
    "place": "未提及",
    "begin_time": "未提及",
    "end_time": "未提及",
    "duration": "未提及"
  }
]
```

同样地，文本中没有提到车辆维修人员平如愿的维修结束时间和维修持续时间，以及维修助手熊深豪的维修地点、维修开始时间、维修结束时间和维修持续时间，所以这些字段都标记为"未提及"或"none"。如有需要，您可以提供更多信息以完善JSON数组中的数据。
"""
))