from pathlib import Path

from experiment.operations.genai.prompt import match_or_register_prompt
import dotenv
import mlflow

dotenv.load_dotenv()

class System:
    # load operations
    # basic predict method (calls operations in sequence)

class Experiment:
    def __init__(self, model_dir=None, prompts_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent

        if prompts_dir is None:
            prompts_dir = model_dir

        self.prompts = [
            match_or_register_prompt(prompt_dir) 
            for prompt_dir in prompts_dir.iterdir() if prompt_dir.is_dir()
        ]

    def chat_complete():
        pass

    def predict():
        pass

