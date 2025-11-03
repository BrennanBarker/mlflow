import openai
from mlflow.pyfunc.utils import pyfunc 
from pydantic import BaseModel
import mlflow
import dotenv

dotenv.load_dotenv()

client = openai.Client()

class LetterCountInput(BaseModel):
    word: str
    letter: str

model = 'gpt-4o'
prompt_uri = 'prompts:/skittish-fawn-303/1'
prompt = mlflow.genai.load_prompt(prompt_uri)

@pyfunc
def predict(model_input: list[LetterCountInput]) -> list[str]:
    predictions = []
    for example in model_input:
        response = client.chat.completions.create(
            model=model,
            messages=prompt.format(**example.model_dump()),
            response_format=prompt.response_format
        )
        predictions.append(response.choices[0].message.content)
    return predictions

mlflow.models.set_model(predict)