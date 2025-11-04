import argparse
import json
from pathlib import Path
import hashlib
import logging

import dotenv
import mlflow
from mlflow.utils.name_utils import _generate_random_name
import yaml

logging.basicConfig(level=logging.INFO)

def get_prompt_hash(dir_path: Path) -> str:
    hash_md5 = hashlib.md5()
    for file_path in sorted(dir_path.rglob('*')):
        if file_path.name in ['template.txt', 'template.json', 'response_format.json'] and file_path.is_file():
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_template(template_path: Path) -> str:
    if template_path.suffix == '.txt':
        return [{"role": "user", "content": template_path.read_text()}]
    elif template_path.suffix == '.json':
        return json.loads(template_path.read_text())
    else:
        raise ValueError("Unsupported template file format. Use .txt or .json")

def get_response_format(response_format_path: Path) -> dict:
    response_schema = json.loads(response_format_path.read_text())
    return {
        "type": "json_schema",
        "strict": True,
        "schema": response_schema  # CHECK: Do I need to drop the $schema field?
    }

def get_matching_prompt(dir_path, hash: str):
    try:
        return mlflow.genai.load_prompt(f'prompts:/{dir_path.name}@{hash}')
    except Exception as e:
        return None

def register_prompt(dir_path: Path):
    metadata_path = dir_path / 'metadata.yaml'
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    name = metadata.get('name') or _generate_random_name()

    tags = {
        **metadata.get('tags', {}),
        'hash': get_prompt_hash(dir_path=dir_path)
    }

    response_format_path = dir_path / 'response_format.json'
    if response_format_path.exists():
        response_format = get_response_format(response_format_path)
    else:
        response_format = None

    template_path = next(dir_path.glob('template.*'), None)
    if template_path is None:
        raise FileNotFoundError("No template file found in the specified directory.")
    template = get_template(template_path)

    registered_prompt = mlflow.genai.register_prompt(
        name=name,
        template=template,
        response_format=response_format,
        tags=metadata.get('tags'),
        commit_message=metadata.get('commit_message'),
    )
    mlflow.genai.set_prompt_alias(name=registered_prompt.name, version=registered_prompt.version, alias=hash)
    return registered_prompt

def match_or_register_prompt(dir_path: Path):
    hash = get_prompt_hash(dir_path)
    hash_matches = get_matching_prompt(hash)
    if hash_matches:
        logging.info(f"Prompt already registered: {hash_matches.uri}")
        return hash_matches
    registered_prompt = register_prompt(dir_path)
    logging.info(f"Registered prompt: {registered_prompt.uri}")


if __name__ == '__main__':
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    args = parser.parse_args()
    match_or_register_prompt(args.dir)