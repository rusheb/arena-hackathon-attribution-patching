def generate_one_token(model, prompt):
    return model.generate(prompt, max_new_tokens=1, top_k=1)
