import json
import logging
from pathlib import Path

import mlflow

from experiment.utils import hash_files

logging.basicConfig(level=logging.INFO)

def get_template_file(dir_path: Path) -> Path:
    valid_filenames = ['template.json', 'template.txt']
    for path in valid_filenames:
        if (dir_path / path).is_file():
            return dir_path / path
    raise FileNotFoundError("No template file found in the specified directory.")

def get_template(template_path: Path) -> str:
    if template_path.suffix == '.txt':
        return [{"role": "user", "content": template_path.read_text()}]
    return json.loads(template_path.read_text())

def get_response_format_file(dir_path: Path) -> Path:
    response_format_path = dir_path / 'response_format.json'
    if response_format_path.is_file():
        return response_format_path
    return None
    
def get_response_format(response_format_path: Path) -> dict:
    response_schema = json.loads(response_format_path.read_text())
    return {
        "type": "json_schema",
        "strict": True,
        "schema": response_schema  # CHECK: Do I need to drop the $schema field?
    }
    
def get_registered_prompt(dir_path: Path):
    prompt_name = dir_path.name

    commit_message_prompt = f'Enter a commit message for prompt {prompt_name}: '
    commit_message = input(commit_message_prompt) or None

    template_file = get_template_file(dir_path)
    template = get_template(template_file)

    response_format_file = get_response_format_file(dir_path)
    if response_format_file:
        response_format = get_response_format(response_format_file)
        prompt_files = [template_file, response_format_file]
    else:
        response_format = None
        prompt_files = [template_file]
    hash = hash_files(prompt_files)

    try:
        registered_prompt = mlflow.genai.load_prompt(f'prompts:/{prompt_name}@{hash}')
        logging.info(f"Prompt already registered: {registered_prompt.uri}")
    except Exception:
        registered_prompt = mlflow.genai.register_prompt(
            name=prompt_name,
            template=template,
            response_format=response_format,
            commit_message=commit_message,
        )   
        logging.info(f"Registered prompt: {registered_prompt.uri}")
    return registered_prompt