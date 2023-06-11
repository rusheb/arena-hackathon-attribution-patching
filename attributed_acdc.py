#%%
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
# chapter = r"chapter1_transformers"
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
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
        center_unembed=True,
        center_writing_weights=True,
      fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
#%%
hook_filter = lambda name: name.endswith("ln1.hook_normalized") or name.endswith("attn.hook_result")
def get_3_caches(model, clean_input, corrupted_input, metric):
    # cache the activations and gradients of the clean inputs
    model.reset_hooks()
    clean_cache = {}
    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()
    model.add_hook(hook_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}
    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()
    model.add_hook(hook_filter, backward_cache_hook, "bwd")

    value = metric(model(clean_input))
    value.backward()
    
    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}
    def forward_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach()
    model.add_hook(hook_filter, forward_cache_hook, "fwd")
    model(corrupted_input)
    
    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache

#%%
def split_layers_and_heads(act: Tensor, model: HookedTransformer) -> Tensor:
    return einops.rearrange(act, '(layer head) batch seq d_model -> layer head batch seq d_model',
                            layer=model.cfg.n_layer,
                            head=model.cfg.n_heads)
# %%
def acdc_brrr(model: HookedTransformer,
              clean_input: Tensor,
              corrupted_input: Tensor,
              metric: Callable[[Tensor], Tensor],
              threshold: float):
    # get the 2 fwd and 1 bwd caches; cache "normalized" and "result" of attn layers
    clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(model, clean_input, corrupted_input, metric)

    # take all pairs of heads, 
    # edges = [
    #       ((layer_sender, head_sender), (layer_receiver, head_receiver))
    #       for layer_sender, layer_receiver in itertools.product(range(model.cfg.n_layer), repeat=2)
    #       for head_sender, head_receiver in itertools.product(range(model.cfg.n_heads), repeat=2)
    #       if layer_sender < layer_receiver
    # ]

    # compute first-order Taylor approximation for each pair to get the attribution of the edge
    clean_head_act = clean_cache.stack_head_results()
    corr_head_act = corrupted_cache.stack_head_results()
    clean_grad_act = clean_grad_cache.stack_head_results()

    # A(C) - A(R)
    head_diff_act = clean_head_act - corr_head_act

    # separate layers and heads
    head_diff_act = split_layers_and_heads(head_diff_act, model)
    clean_grad_act = split_layers_and_heads(clean_grad_act, model)
    
    # compute the attribution of the path
    path_attr = einops.einsum(
        head_diff_act, clean_grad_act,
        'layer_start head_start batch seq d_model, layer_end head_end batch seq d_model -> layer_start head_start layer_end head_end',
    )

    # do "causal masking" to make sure we don't have the sender after receiver
    correct_layer_order_mask = (
        t.arange(model.cfg.n_layers)[:, None, None, None] < 
        t.arange(model.cfg.n_layers)[None, None, :, None]).to(path_attr.device)
    null_val = t.full(1, float('inf'), device=path_attr.device)
    path_attr = t.where(correct_layer_order_mask, path_attr, null_val)

    # prune all pairs whose attribution is below the threshold
    should_prune = path_attr < threshold
    pruned_model = model.copy()
    for layer_sender, layer_receiver in itertools.product(range(model.cfg.n_layer), repeat=2):
        for head_sender, head_receiver in itertools.product(range(model.cfg.n_heads), repeat=2):
            if should_prune[layer_sender, head_sender, layer_receiver, head_receiver]:
                # prune the edge
                #! TODO

    # compute metric on pruned subgraph vs whole graph
    metric_full_graph = metric(model(clean_input))
    metric_pruned_graph = metric(pruned_model(clean_input))

    # return the pruned subgraph and the metrics
    return pruned_model, metric_full_graph, metric_pruned_graph
    
# %%