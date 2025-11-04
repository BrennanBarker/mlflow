from pathlib import Path
import mlflow
import dotenv 
import openai
import yaml
import importlib

from register_prompt import match_or_register_prompt

dotenv.load_dotenv()

mlflow.set_tracking_uri('http://localhost:5000')

system_dir = Path('./system')
dataset_path = Path('./data.jsonl')
inputs_cols = ['text']
expectations_cols = ['letter_count']
scorers_file = 'scorers.py'

model_dirs = system_dir / 'models'

import importlib.util
spec= importlib.util.spec_from_file_location('predict', Path('system') / 'predict.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

    
mlflow.set_experiment('letter counting')
mlflow.openai.autolog()
with mlflow.start_run():
    model_configs = []
    all_prompts = []
    for model_dir in model_dirs.iterdir():
        if model_dir.is_dir():
            prompts = []
            for prompt_dir in (model_dir / 'prompts').iterdir():
                prompt = match_or_register_prompt(prompt_dir)
                prompts.append(prompt)
            with open(model_dir / 'config.yaml') as f:
                model_config = yaml.safe_load(f)
            model_configs.append(model_config)
            all_prompts.append(prompts)
            
            mlflow.openai.log_model(
                task=model_config.task,
                model=model_config.model,
                prompts=prompts,
                params=model_config.get('params')
            )
    predict_fn = module.make_predict_fn(model_configs, all_prompts)
    mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=to_eval_dataset(dataset, inputs_cols, expectations_cols),
        scorers=
    )

print(f"Registered model: {info.model_uri}")