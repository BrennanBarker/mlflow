import pandas as pd
import mlflow
import openai
import dotenv
from mlflow.pyfunc.utils import pyfunc 
from pydantic import BaseModel

dotenv.load_dotenv()

experiment_name = 'letter counting'
dataset_path = 'data.jsonl'
prompt_uri = 'prompts:/skittish-fawn-303/1'
model = 'gpt-4o'

mlflow.openai.autolog()
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    data = pd.read_json(dataset_path, lines=True)
    ds = mlflow.data.from_pandas(data)
    mlflow.log_input(ds, 'letter_counting_data')
    mlflow.log_
    prompt = mlflow.genai.load_prompt(prompt_uri)
    all_predictions = []
    examples = data.to_dict(orient='records')
    for example in examples:        
        predictions = predict([example])
        all_predictions.extend(predictions)

