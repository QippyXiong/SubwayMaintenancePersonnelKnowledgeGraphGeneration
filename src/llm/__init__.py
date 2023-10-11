
from pathlib import Path

from . import gpt

from .prompt_info_extraction import gpt_maintainance_record_extraction

# from database.utils import GenerateCap

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

