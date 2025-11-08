from pathlib import Path

import openai
from pydantic import BaseModel
import yaml

from experiment.operations.genai.prompt import get_registered_prompt

class ChatCompletionParams(BaseModel):
    temperature: float
    max_completion_tokens: int
    top_p: int

class ChatCompletionConfig(BaseModel):
    model: str
    # network: str
    params: ChatCompletionParams

    @classmethod
    def from_yaml(cls, fp: Path):
        with open(fp) as f:
            obj = yaml.safe_load(f)
            return cls.model_validate(obj)

class ChatCompletionOperation():
    def __init__(self, dir_path: str):
        dir_path = Path(dir_path)
        config_path = dir_path / 'config.yaml'
        self.config = ChatCompletionConfig.from_yaml(config_path)
        self.prompt = get_registered_prompt(dir_path)
        self.client = openai.Client(
            # api_key = ...
            # base_url= ...
        )

    def execute(self, **inputs):
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.prompt.format(**inputs),
            response_format=self.prompt.response_format,
            **self.config.params.model_dump()
        )
        return response.choices[0].message['content']
        