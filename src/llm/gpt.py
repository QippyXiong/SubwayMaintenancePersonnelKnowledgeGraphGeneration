import openai
from openai.openai_object import OpenAIObject
from typing import Callable


def connect_openai(api_key: str, api_base: str):
	openai.api_key = api_key
	openai.api_base = api_base


def chat(content: str, num_choices: int = 1, role: str = "user") -> list[str]:
	r"""
	
	"""
	res = openai.ChatCompletion.create(
		model='gpt-3.5-turbo-0613',
		messages=[{
			'role': role,
			'content': content
		}],
		stream=False,
		n=num_choices
	)
	return [ c['message']['content'] for c in res['choices'] ]


def chat_stream(content: str, stream_handler: Callable[[OpenAIObject], None], num_choices: int = 1, role='user') -> None:
	res = openai.ChatCompletion.create(
		model='gpt-3.5-turbo',
		messages=[{
			'role': role,
			'content': content
		}],
		stream=True,
		n=num_choices
	)
	for event in res:
		r"""
		{
		"id": "chatcmpl-86H1dMtMox5OhUqLT0f9oPsfI3r5H",
		"object": "chat.completion.chunk",
		"created": 1696506105,
		"model": "gpt-3.5-turbo-0301",
		"choices": [
			{
			"index": 0,
			"delta": {},
			"finish_reason": "stop"
			}
		]
		}
		"""
		print('stream recieve', event)
		event: OpenAIObject
		stream_handler(event)
