# %%
from transformer_lens import HookedTransformer

from attribution_patching.utils import generate_one_token

print("Generating text...")

model = HookedTransformer.from_pretrained("gpt2-small")
prompt = "Attention is all you"
continuation = generate_one_token(model, prompt)

print(continuation)

# %%
