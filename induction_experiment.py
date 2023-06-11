#%%
import os
import sys
import time
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

from attributed_acdc import acdc_nodes, format_heads_pruned

# Make sure exercises are in the path
# chapter = r"chapter1_transformers"
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

# from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
# from part1_transformer_from_scratch.solutions import get_log_probs
# import part2_intro_to_mech_interp.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

#%%
def create_model() -> HookedTransformer:
    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True, # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b", 
        seed=398,
        use_attn_result=True,
        normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer"
    )
    weights_dir = (Path(os.getcwd()) / "temp/model_weights/attn_only_2L_half.pth").resolve()

    if not weights_dir.exists():
        url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
        output = str(weights_dir)
        gdown.download(url, output)
    
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_dir, map_location=device)
    model.load_state_dict(pretrained_weights)

    return model
# %%
def random_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Int[Tensor, "batch seq_len"]:
    '''
    Generates a sequence of random tokens

    Outputs are:
        rand_tokens: [batch, seq_len]
    '''
    rand_tokens = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64).to(device)
    return rand_tokens

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    # SOLUTION
    rep_tokens_half = random_tokens(model, seq_len, batch)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1)
    return rep_tokens

# %%
if MAIN:
    model = create_model()
    clean_input = generate_repeated_tokens(model, 63, 64)
    corr_input = t.cat([clean_input[:, :64], random_tokens(model, 63, 64)], dim=1)

def induction_metric(logits: Float[Tensor, 'batch seq_len d_vocab']) -> Float[Tensor, 'batch']:
    '''
    Computes the induction metric, which is the average logit of the target
    token in the second half of the sentence where we're repeating tokens
    '''
    batch, seq_len, d_vocab = logits.shape

    # Create indices for indexing into logits and clean_input
    indices = t.arange(seq_len)[None, :].expand(batch, seq_len)

    # Index into logits using clean_input to directly get the target_logits
    target_logits = logits[t.arange(batch)[:, None], indices, clean_input]

    return target_logits.mean().to(device)
# %%
if MAIN:
    st = time.time()
    pruned_model, should_prune = acdc_nodes(model, clean_input, corr_input,
                                            induction_metric, 0.1,
                                            create_model=create_model)

    print(f"Time taken: {time.time() - st:.2f}s")
    print(f"Number of heads pruned: {should_prune.sum()}, out of {should_prune.numel()}")
    print(f"Nodes that weren't pruned: {format_heads_pruned(should_prune.logical_not())}")
# %%
# print out the attention patterns of the heads that weren't pruned
if MAIN:
    tokens = clean_input[0]
    layer = 1
    logits, cache = model.run_with_cache(tokens)
    display(cv.attention.attention_patterns(
        tokens=model.to_str_tokens(tokens),
        attention=cache['pattern', layer, 'attn'][0],
        attention_head_names=[f'L{layer}H{head}' for head in range(model.cfg.n_heads)],
    ))
# %%
