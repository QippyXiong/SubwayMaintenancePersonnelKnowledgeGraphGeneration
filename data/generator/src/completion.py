import openai
import json
import time

class Completion:
    def __init__(self) -> None:
        with open("data/dataset/generator/OPENAI_APIKEY.txt", "r", encoding="ASCII") as fp:
            openai.api_key = fp.readline()
        
        with open("data/dataset/generator/PromptConfig.json", "r", encoding="ASCII") as fp:
            self.config = json.loads("".join(fp.readlines()))
            print(str(self.config))
    
    def prompt(self, prompt: str) -> tuple[str, dict]:
        if(prompt):
            self.config["prompt"] = prompt
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
            ],
            temperature=0,
        )
        # calculate the time it took to receive the response
        response_time = time.time() - start_time

        # print the time delay and text received
        print(f"Full response received {response_time:.2f} seconds after request")
        print(f"Full response received:\n{response}")

    