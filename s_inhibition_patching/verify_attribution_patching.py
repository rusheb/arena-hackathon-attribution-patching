# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP


# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

sys.path.append('..')
from attribution_patching import neel_plotly
from attribution_patching.plotly_utils import imshow, line, scatter, bar
# import part3_indirect_object_identification.tests as tests

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

MAIN = __name__ == "__main__"
# %%
if MAIN:
	model = HookedTransformer.from_pretrained(
		"gpt2-small",
	)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

# %%
prompts = ['When John and Mary went to the shops, John gave the bag to', 'When John and Mary went to the shops, Mary gave the bag to', 'When Tom and James went to the park, James gave the ball to', 'When Tom and James went to the park, Tom gave the ball to', 'When Dan and Sid went to the shops, Sid gave an apple to', 'When Dan and Sid went to the shops, Dan gave an apple to', 'After Martin and Amy went to the park, Amy gave a drink to', 'After Martin and Amy went to the park, Martin gave a drink to']
answers = [(' Mary', ' John'), (' John', ' Mary'), (' Tom', ' James'), (' James', ' Tom'), (' Dan', ' Sid'), (' Sid', ' Dan'), (' Martin', ' Amy'), (' Amy', ' Martin')]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(clean_tokens)) ]
    ]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = t.tensor([[model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))], device=model.cfg.device)
print("Answer token indices", answer_token_indices)

# %%
receiver_heads = [(8,6), (8,10), (7,9), (7,3)]

def logits_to_ave_logit_diff_2(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset = ioi_dataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

with t.no_grad():
  ioi_logits_original = model(ioi_dataset.toks)
  abc_logits_original = model(abc_dataset.toks)

ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

def ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = ioi_average_logit_diff,
    corrupted_logit_diff: float = abc_average_logit_diff,
    ioi_dataset: IOIDataset = ioi_dataset,
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

# %%
filter_is_val = lambda name: name.endswith("hook_v") or name.endswith("hook_v_input") or name.endswith("hook_z") or name.endswith("ln1.hook_scale") 
def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()
    model.add_hook(filter_is_val, forward_cache_hook, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()
    model.add_hook(filter_is_val, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)

clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(model, ioi_dataset.toks, ioi_metric)
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, abc_dataset.toks, ioi_metric)

# %%

HEAD_NAMES = [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
HEAD_NAMES_QKV = [f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["V"]]

# def get_head_vector_grad_input_from_grad_cache(
#         grad_cache: ActivationCache, 
#         activation_name: Literal["q", "k", "v"],
#         layer: int
#     ) -> Float[Tensor, "batch pos head_index d_model"]:
#     vector_grad = grad_cache[activation_name + "_input", layer]
#     return vector_grad

def get_head_vector_grad_input_from_grad_cache(grad_cache: ActivationCache, 
        activation_name: Literal["q", "k", "v"],
        layer: int
    ) -> Float[Tensor, "batch pos head_index d_model"]:
    vector_grad = grad_cache[activation_name, layer]
    ln_scales = grad_cache["scale", layer, "ln1"]
    attn_layer_object = model.blocks[layer].attn
    if activation_name == "q":
        W = attn_layer_object.W_Q
    elif activation_name == "k":
        W = attn_layer_object.W_K
    elif activation_name == "v":
        W = attn_layer_object.W_V
    else:
        raise ValueError("Invalid activation name")
    print(vector_grad.shape, ln_scales.shape, W.shape)
    return einops.einsum(
         vector_grad, 
         ln_scales.squeeze(), 
         W,
         "batch pos head_index d_head, batch pos head_index, head_index d_model d_head -> batch pos head_index d_model"
        )

def get_stacked_head_vector_grad_input(grad_cache, activation_name: Literal["q", "k", "v"]) -> Float[Tensor, "layer batch pos head_index d_model"]:
    return t.stack([get_head_vector_grad_input_from_grad_cache(grad_cache, activation_name, l) for l in range(model.cfg.n_layers)], dim=0)

def get_full_vector_grad_input(grad_cache) -> Float[Tensor, "qkv layer batch pos head_index d_model"]:
    return t.stack([get_stacked_head_vector_grad_input(grad_cache, activation_name) for activation_name in ['v']], dim=0) # ['q', 'k', 'v']

def attr_patch_head_path(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache
    ) -> Float[Tensor, "qkv dest_component src_component pos"]:
    """
    Computes the attribution patch along the path between each pair of heads.

    Sets this to zero for the path from any late head to any early head

    """
    start_labels = HEAD_NAMES
    end_labels = HEAD_NAMES_QKV
    full_vector_grad_input = get_full_vector_grad_input(corrupted_grad_cache)
    clean_head_result_stack = clean_cache.stack_head_results(-1)
    corrupted_head_result_stack = corrupted_cache.stack_head_results(-1)
    diff_head_result = einops.rearrange(
        clean_head_result_stack - corrupted_head_result_stack,
        "(layer head_index) batch pos d_model -> layer batch pos head_index d_model",
        layer = model.cfg.n_layers,
        head_index = model.cfg.n_heads,
    )
    path_attr = einops.einsum(
         full_vector_grad_input, 
         diff_head_result,
         "qkv layer_end batch pos head_end d_model, layer_start batch pos head_start d_model -> qkv layer_end head_end layer_start head_start pos", 
        )
    correct_layer_order_mask = (
        t.arange(model.cfg.n_layers)[None, :, None, None, None, None] > 
        t.arange(model.cfg.n_layers)[None, None, None, :, None, None]).to(path_attr.device)
    zero = t.zeros(1, device=path_attr.device)
    path_attr = t.where(correct_layer_order_mask, path_attr, zero)

    path_attr = einops.rearrange(
        path_attr,
        "qkv layer_end head_end layer_start head_start pos -> (layer_end head_end qkv) (layer_start head_start) pos",
    )
    return path_attr, end_labels, start_labels

head_path_attr, end_labels, start_labels = attr_patch_head_path(clean_cache, corrupted_cache, corrupted_grad_cache)
neel_plotly.imshow(head_path_attr.sum(-1), y=end_labels, yaxis="Path End (Head Input)", x=start_labels, xaxis="Path Start (Head Output)", title="Head Path Attribution Patching")
# %%
head_path_attr_by_layer = einops.rearrange(
    head_path_attr,
    "(layer_end head_end) (layer_start head_start) pos -> layer_end head_end layer_start head_start pos",
    layer_end = model.cfg.n_layers,
    layer_start = model.cfg.n_layers,
    head_end = model.cfg.n_heads,
    head_start = model.cfg.n_heads,
)
head_path_attr_s_inhb_by_head = t.stack([
    head_path_attr_by_layer[l, h, :8, :, :] for l, h in receiver_heads
]) # (receiver_head, layer_start head_start pos)
head_path_attr_s_inhb = head_path_attr_s_inhb_by_head.sum((0, -1)) # (layer_start head_start)
neel_plotly.imshow(head_path_attr_s_inhb, xaxis="Head", yaxis="Layer", title="Head Path Attribution Patching (S Inhibition)")
# %%
