import mlflow
import dotenv 
import openai

dotenv.load_dotenv()

mlflow.set_tracking_uri('http://localhost:5000')

mlflow.openai.autolog()

mlflow.search_logged_models(['letter_counting_model'])
mlflow.set_experiment('letter counting')
with mlflow.start_run():
    # info = mlflow.openai.log_model(
    #     task=openai.chat.completions,
    #     model='gpt-4o',
    #     prompts=['prompts:/skittish-fawn-303/1'],
    # )


    info = mlflow.pyfunc.log_model(
        name='letter_counting_model',
        python_model='model.py',
        input_example=[{"word": "hello", "letter": "l"}],
        prompts=['prompts:/skittish-fawn-303/1'],
    )   
print(f"Registered model: {info.model_uri}")