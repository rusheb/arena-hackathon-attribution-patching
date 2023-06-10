
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

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

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

# %%

from ioi_dataset import NAMES, IOIDataset
# %%
N = 25
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device)
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
# %%
# %%

def format_prompt(sentence: str) -> str:
	'''Format a prompt by underlining names (for rich print)'''
	return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
	'''Makes and displays a table, from cols rather than rows (using rich print)'''
	table = Table(*colnames, title=title)
	rows = list(zip(*cols))
	f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
	for row in rows[:n_rows]:
		table.add_row(*list(map(f, row)))
	rprint(table)

# %%


if MAIN:
	make_table(
		colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
		cols = [
			map(format_prompt, ioi_dataset.sentences), 
			model.to_string(ioi_dataset.s_tokenIDs).split(), 
			model.to_string(ioi_dataset.io_tokenIDs).split(), 
			map(format_prompt, abc_dataset.sentences), 
		],
		title = "Sentences from IOI vs ABC distribution",
	)

# %%

if MAIN:
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



if MAIN:
	model.reset_hooks(including_permanent=True)
	
	ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
	abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)
	
	ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
	abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)
	
	ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
	abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

# %%


if MAIN:
	print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
	print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")
	
	make_table(
		colnames = ["IOI prompt", "IOI logit diff", "ABC prompt", "ABC logit diff"],
		cols = [
			map(format_prompt, ioi_dataset.sentences), 
			ioi_per_prompt_diff,
			map(format_prompt, abc_dataset.sentences), 
			abc_per_prompt_diff,
		],
		title = "Sentences from IOI vs ABC distribution",
	)

# %%

if MAIN:
	def ioi_metric_2(
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



if MAIN:
	print(f"IOI metric (IOI dataset): {ioi_metric_2(ioi_logits_original):.4f}")
	print(f"IOI metric (ABC dataset): {ioi_metric_2(abc_logits_original):.4f}")

# %%

# FLAT SOLUTION NOINDENT NOCOMMENT
def patch_or_freeze_head_vectors(
	orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
	hook: HookPoint, 
	new_cache: ActivationCache,
	orig_cache: ActivationCache,
	head_to_patch: Tuple[int, int], 
) -> Float[Tensor, "batch pos head_index d_head"]:
	'''
	This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them
	to their values in orig_cache), except for head_to_patch (if it's in this layer) which
	we patch with the value from new_cache.

	head_to_patch: tuple of (layer, head)
		we can use hook.layer() to check if the head to patch is in this layer
	'''
	# Setting using ..., otherwise changing orig_head_vector will edit cache value too
	orig_head_vector[...] = orig_cache[hook.name][...]
	if head_to_patch[0] == hook.layer():
		orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
	return orig_head_vector


# FLAT SOLUTION NOINDENT NOCOMMENT
def patch_head_input(
	orig_activation: Float[Tensor, "batch pos head_idx d_head"],
	hook: HookPoint,
	patched_cache: ActivationCache,
	head_list: List[Tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
	'''
	Function which can patch any combination of heads in layers,
	according to the heads in head_list.
	'''
	heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
	orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
	return orig_activation


if MAIN:
	def get_path_patch_head_to_heads(
		receiver_heads: List[Tuple[int, int]],
		receiver_input: str,
		model: HookedTransformer,
		patching_metric: Callable,
		new_dataset: IOIDataset = abc_dataset,
		orig_dataset: IOIDataset = ioi_dataset,
		new_cache: Optional[ActivationCache] = None,
		orig_cache: Optional[ActivationCache] = None,
	) -> Float[Tensor, "layer head"]:
		'''
		Performs path patching (see algorithm in appendix B of IOI paper), with:

			sender head = (each head, looped through, one at a time)
			receiver node = input to a later head (or set of heads)

		The receiver node is specified by receiver_heads and receiver_input.
		Example (for S-inhibition path patching the queries):
			receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
			receiver_input = "v"

		Returns:
			tensor of metric values for every possible sender head
		'''
		model.reset_hooks()

		assert receiver_input in ("k", "q", "v")
		receiver_layers = set(next(zip(*receiver_heads)))
		# model.blocks.0.attn.hook_v
		receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
		receiver_hook_names_filter = lambda name: name in receiver_hook_names

		results = t.zeros(max(receiver_layers), model.cfg.n_heads, device="cuda", dtype=t.float32)
		
		# ========== Step 1 ==========
		# Gather activations on x_orig and x_new

		# Note the use of names_filter for the run_with_cache function. Using it means we 
		# only cache the things we need (in this case, just attn head outputs).
		z_name_filter = lambda name: name.endswith("z")
		if new_cache is None:
			_, new_cache = model.run_with_cache(
				new_dataset.toks, 
				names_filter=z_name_filter, 
				return_type=None
			)
		if orig_cache is None:
			_, orig_cache = model.run_with_cache(
				orig_dataset.toks, 
				names_filter=z_name_filter, 
				return_type=None
			)

		# Note, the sender layer will always be before the final receiver layer, otherwise there will
		# be no causal effect from sender -> receiver. So we only need to loop this far.
		for (sender_layer, sender_head) in tqdm(list(itertools.product(
			range(max(receiver_layers)),
			range(model.cfg.n_heads)
		))):

			# ========== Step 2 ==========
			# Run on x_orig, with sender head patched from x_new, every other head frozen

			hook_fn = partial(
				patch_or_freeze_head_vectors,
				new_cache=new_cache, 
				orig_cache=orig_cache,
				head_to_patch=(sender_layer, sender_head),
			)
			model.add_hook(z_name_filter, hook_fn, level=1)
			
			_, patched_cache = model.run_with_cache(
				orig_dataset.toks, 
				names_filter=receiver_hook_names_filter,  
				return_type=None
			)
			# model.reset_hooks(including_permanent=True)
			assert set(patched_cache.keys()) == set(receiver_hook_names)

			# ========== Step 3 ==========
			# Run on x_orig, patching in the receiver node(s) from the previously cached value
			
			hook_fn = partial(
				patch_head_input, 
				patched_cache=patched_cache, 
				head_list=receiver_heads,
			)
			patched_logits = model.run_with_hooks(
				orig_dataset.toks,
				fwd_hooks = [(receiver_hook_names_filter, hook_fn)], 
				return_type="logits"
			)

			# Save the results
			results[sender_layer, sender_head] = patching_metric(patched_logits)

		return results

# %%


if MAIN:
	model.reset_hooks()
	
	s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
		receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
		receiver_input = "v",
		model = model,
		patching_metric = ioi_metric_2
	)
	
	imshow(
		100 * s_inhibition_value_path_patching_results,
		title="Direct effect on S-Inhibition Heads' values", 
		labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
		width=600,
		coloraxis=dict(colorbar_ticksuffix = "%"),
	)
