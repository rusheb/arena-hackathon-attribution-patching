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
