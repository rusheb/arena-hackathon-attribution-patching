# %%
# Import stuff
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional, Callable, Literal
from jaxtyping import Float
from functools import partial
import copy
import itertools
import json

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML, Markdown
from attribution_patching.neel_plotly import imshow, line

# %%
# import pysvelte

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

# %%
import transformer_lens.patching as patching

# %%
model = HookedTransformer.from_pretrained("gpt2-small")
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
# %%
prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor(
    [
        [model.to_single_token(answers[i][j]) for j in range(2)]
        for i in range(len(answers))
    ],
    device=model.cfg.device,
)
print("Answer token indices", answer_token_indices)


# %%
# Define the metric
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
# %%
CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff


def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )


print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")
# %%
Metric = Callable[[Float[Tensor, "batch_and_pos_dims d_model"]], Float]
# %%
noop_filter = lambda name: True

def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(noop_filter, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(noop_filter, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, clean_tokens, ioi_metric
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, corrupted_tokens, ioi_metric
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))
# %%
[k for k in corrupted_grad_cache.keys() if ".0." in k or "blocks" not in k]

# %%
HEAD_NAMES = [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
HEAD_NAMES_QKV = [f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]]
print(HEAD_NAMES[:5])
print(HEAD_NAMES_SIGNED[:5])
print(HEAD_NAMES_QKV[:5])
# %% NEELS 
def get_head_vector_grad_input_from_grad_cache(
        grad_cache: ActivationCache, 
        activation_name: Literal["q", "k", "v"],
        layer: int
    ):
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

    return einsum("batch pos head_index d_head, batch pos, head_index d_model d_head -> batch pos head_index d_model", vector_grad, ln_scales.squeeze(-1), W)

def get_stacked_head_vector_grad_input(grad_cache, activation_name: Literal["q", "k", "v"]):
    return torch.stack([get_head_vector_grad_input_from_grad_cache(grad_cache, activation_name, l) for l in range(model.cfg.n_layers)], dim=0)

# %% OUR CODE

# def get_stacked_head_vector_grad_input(
#     grad_cache, activation_name: Literal["q", "k", "v"]
# ) -> Float[Tensor, "layer batch pos head_index d_model"]:
#     return torch.stack(
#         [grad_cache[f"{activation_name}_input", l] for l in range(model.cfg.n_layers)],
#         dim=0,
#     )


def get_full_vector_grad_input(
    grad_cache,
) -> Float[Tensor, "qkv layer batch pos head_index d_model"]:
    return torch.stack(
        [
            get_stacked_head_vector_grad_input(grad_cache, activation_name)
            for activation_name in ["q", "k", "v"]
        ],
        dim=0,
    )


def attr_patch_head_path(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
):
    start_labels = HEAD_NAMES
    end_labels = HEAD_NAMES_QKV
    full_vector_grad_input = get_full_vector_grad_input(corrupted_grad_cache)
    clean_head_result_stack = clean_cache.stack_head_results(-1)
    corrupted_head_result_stack = corrupted_cache.stack_head_results(-1)
    diff_head_result = einops.rearrange(
        clean_head_result_stack - corrupted_head_result_stack,
        "(layer head_index) batch pos d_model -> layer batch pos head_index d_model",
        layer=model.cfg.n_layers,
        head_index=model.cfg.n_heads,
    )
    path_attr = einsum(
        "qkv layer_end batch pos head_end d_model, \
         layer_start batch pos head_start d_model -> \
         qkv layer_end head_end layer_start head_start pos", 
        full_vector_grad_input, 
        diff_head_result)
    correct_layer_order_mask = (
        torch.arange(model.cfg.n_layers)[None, :, None, None, None, None] > 
        torch.arange(model.cfg.n_layers)[None, None, None, :, None, None]).to(path_attr.device)
    zero = torch.zeros(1, device=path_attr.device)
    path_attr = torch.where(correct_layer_order_mask, path_attr, zero)

    path_attr = einops.rearrange(
        path_attr,
        "qkv layer_end head_end layer_start head_start pos -> (layer_end head_end qkv) (layer_start head_start) pos",
    )
    return path_attr, end_labels, start_labels

head_path_attr, end_labels, start_labels = attr_patch_head_path(clean_cache, corrupted_cache, corrupted_grad_cache)
# %%
imshow(head_path_attr.sum(-1), y=end_labels, yaxis="Path End (Head Input)", x=start_labels, xaxis="Path Start (Head Output)", title="Head Path Attribution Patching")


#%% 
def attr_patch_head_out(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache,
    ) -> Float[Tensor, "component pos"]:
    labels = HEAD_NAMES

    clean_head_out = clean_cache.stack_head_results(-1, return_labels=False)
    corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False)
    corrupted_grad_head_out = corrupted_grad_cache.stack_head_results(-1, return_labels=False)
    head_out_attr = einops.reduce(
        corrupted_grad_head_out * (clean_head_out - corrupted_head_out),
        "component batch pos d_model -> component pos",
        "sum"
    )
    return head_out_attr, labels

head_out_attr, head_out_labels = attr_patch_head_out(clean_cache, corrupted_cache, corrupted_grad_cache)
# imshow(head_out_attr, y=head_out_labels, yaxis="Component", xaxis="Position", title="Head Output Attribution Patching")
# sum_head_out_attr = einops.reduce(head_out_attr, "(layer head) pos -> layer head", "sum", layer=model.cfg.n_layers, head=model.cfg.n_heads)
# imshow(sum_head_out_attr, yaxis="Layer", xaxis="Head Index", title="Head Output Attribution Patching Sum Over Pos")


# %%
head_out_values, head_out_indices  = head_out_attr.sum(-1).abs().sort(descending=True)
line(head_out_values)
top_head_indices = head_out_indices[:22].sort().values
top_end_indices = []
top_end_labels = []
top_start_indices = []
top_start_labels = []
for i in top_head_indices:
    i = i.item()
    top_start_indices.append(i)
    top_start_labels.append(start_labels[i])
    for j in range(3):
        top_end_indices.append(3*i+j)
        top_end_labels.append(end_labels[3*i+j])

imshow(head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1), y=top_end_labels, yaxis="Path End (Head Input)", x=top_start_labels, xaxis="Path Start (Head Output)", title="Head Path Attribution Patching (Filtered for Top Heads)")

# %%
for j, composition_type in enumerate(["Query", "Key", "Value"]):
    imshow(head_path_attr[top_end_indices, :][:, top_start_indices][j::3].sum(-1), y=top_end_labels[j::3], yaxis="Path End (Head Input)", x=top_start_labels, xaxis="Path Start (Head Output)", title=f"Head Path to {composition_type} Attribution Patching (Filtered for Top Heads)")
# %%
for j, composition_type in enumerate(["Query", "Key", "Value"]):
    imshow(head_path_attr[top_end_indices, :][:, top_start_indices][j::3].sum(-1), y=top_end_labels[j::3], yaxis="Path End (Head Input)", x=top_start_labels, xaxis="Path Start (Head Output)", title=f"Head Path to {composition_type} Attribution Patching (Filtered for Top Heads)")
# %%
