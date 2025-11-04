import openai

client = openai.OpenAI()

def make_predict_fn(model_configs, prompts):
    model_config = model_configs[0]
    prompt = prompts[0]
    def predict_fn(**inputs):
        response = client.chat.completions.create(
            model=model_config.model,
            messages=prompt.format(**inputs),
            response_format=prompt.response_format,
            **model_config.params
        )
        return response.choices[0].message.content
    return predict_fn