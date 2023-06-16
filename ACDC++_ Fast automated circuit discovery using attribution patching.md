# ACDC++: Fast automated circuit discovery using attribution patching
## Abstract
Recent advancements in automated circuit discovery have made it possible to find circuits responsible for particular behaviors without much human intervention. However, existing algorithms require a relatively large number of forward passes through the model, which means they are quite slow to run. We integrate these algorithms with the first-order Taylor approximations proposed by Neel Nanda, arriving at a much faster algorithm. Our final algorithm takes a model, clean dataset, corrupted dataset, and a metric evaluating the model's behavior on the given task, and returns a pruned model and a list of heads in the circuit implementing the task. Most importantly, the algorithm is faster than ACDC, discovering the IOI circuit in GPT2-small in 4.1 seconds compared to ACDC's 8 minutes.

## Goal

**Develop an algorithm which speeds up automated circuit discovery using attribution patching.**

### Motivation

Automated Circuit DisCovery (ACDC) is a technique which aims to automatically identify important units of a neural network in order to speed up the Mechanistic Interpretability workflow for finding circuits. The technique has had success finding many circuits, including those for IOI, induction, docstring, greater than, and tracr.

However, the ACDC technique is currently too computationally intensive to apply to models the size of GPT2-XL or larger. This is because multiple forward passes are required for each edge in the computational graph. 

Our aim was to create a modified algorithm which speeds up ACDC using the extremely fast Attribution Patching technique.

### Desiderata for the algorithm

- Accurately find circuits which have been found manually
- Work arbitrary tasks on arbitrarily large models
- Require a constant number of forward/backward passes of the model
    
    
## Related work

### Attribution Patching
- Attribution patching is a technique which uses gradients to take a linear approximation to activation patching.
- This reduces the number of passes from linear in number of activations patched, to constant
- The approximation is more valid when patching small activations than large ones. 
![](https://hackmd.io/_uploads/r1OKBuQPn.png)


### ACDC
![](https://hackmd.io/_uploads/Sytz3dQP2.png)
(source: ACDC Paper - Conmy et. al. 2023)

- ACDC is an algorithm which uses iterated pruning to find important subgraphs of a computational graph representing a neural network.


## Results

<!-- We test whether attribution patching is an actual  -->

### Summary
1. Replicated detection of S-inhibition heads from the IOI paper, using attribution patching instead of activation patching.
1. Designed and implemented algorithm that identifies the IOI subgraph at least 100x faster than ACDC


### 1. Identification of S-inhibition heads

We showed that attribution patching was able to identify the important heads in the IOI circuit by replicating figure 5(b) from the IOI paper

**Original Direct Effect on S-Inhibition Head's Values**
![](https://hackmd.io/_uploads/SkiHOFmv2.png)
**Direct Effect on S-Inhibition Head's Values using Attribution Patching**
![](https://hackmd.io/_uploads/rylsBuYQPn.png)


### 2. Circuit Detection Algorithm
We designed and implemented an algorithm which identified the important heads in the IOI circuit in under **4.1 seconds** on an A10 GPU. This is compared to 8 minutes for the ACDC algorithm running on an A100.
<!-- *TODO: Image or code?* -->
**Nodes in the IOI circuit in GPT2-small that our algorithm finds for a given threshold value**
![Nodes in the IOI circuit depending on the threshold value](https://hackmd.io/_uploads/BkjS_t7Dh.png)

The original IOI paper identifies 26 heads relevant to the task. Exploring a few of these thresholds:
- For threshold 0.2, our method identifies 33 heads, of which 18 are part of the original IOI paper. Our method did not pick up on 8 heads, of which only 3 were not Backup Name Mover heads (BNMs). We believe this distinction is relevant since BNMs do not play a large role in the model's computation unless the regular name mover heads are ablated, and thus we would not expect our algorithm to detect them.
- For threshold 0.3, our method identifies 21 head, of which 16 are heads also identified by the IOI paper. Our method did not pick up on 10 heads, of which only 5 were not Backup Name Mover heads.
- For threshold 0.4, our method identifies 16 heads, of which 14 are also identified by the original IOI paper. Our method did not pick up on 12 heads, of which 5 were not Backup Name Mover heads. 1 of the 5 heads was a Negative Name Mover head.
- For threshold 0.5, our method identifies 15 heads, of which all 15 are also identified by the IOI paper. Our method did not pick up on the remaining 15 heads, of which 10 are not Backup Name Mover heads. 1 of the 10 heads is a Negative Name Mover head.

We seem to miss backup name mover heads, but capture the negative name move heads -- this is to be expected as backup name mover heads do not significantly contribute unless parts of the model are ablated, while negative heads always contribute to the outcome. Layer 6 head 0 is also consistently falsely identified as part of the circuit, and a case study on why would be worth exploring.

#### Method: Node attribution patching 
<!-- (first order approximation of Activation Patching) -->
- Two forward passes: caching activations for clean and corrupted prompt, yielding `clean_cache, corrupted_cache`
- One backward pass, caching gradients of the loss metric on the clean prompt w.r.t head activations 
- Compute the importance of each node in the computational graph using attribution patching: `(clean_head_activations - corr_head_activations) * clean_head_grad_in`
- Do ACDC-style thresholding based on the metric (eg. logit difference for the IOI task)
- Prune nodes by filling their `W_Q` with zeros (we believe this is vaguely analogous to mean-pruning since it effectively turns the QK circuit into an averaging circuit, but without taking the activations too far out of distribution since the OV circuit stays intact; however we still need to verify this in future work by comparing it to resample-pruning)

___
**Algorithm 1: The node-based automated circuit discovery using attribution patching**

**Data**: Model $M$, clean dataset $C$, corrupted dataset $R$, metric $F:\mathcal{L}\to\mathbb{R}$, and threshold $\tau\in\mathbb{R}$
**Result**: Circuit (subgraph) of the model $T\subseteq M$, and list of booleans indicating which attention heads were pruned $P\in\{0,1\}^n$<!-- Let $M_\text{cache}$ be a function that takes an input, and returns the cached activations of running that input through model $M$. Similarly, let $M_{\nabla\text{cache}}$ be a function that takes an input and a metric, and returns the cached gradients of the metric with respect to the attention head activations during the backward pass. -->
$A_C\leftarrow M_\text{cache}(C)\qquad\qquad\qquad$ // Cache activations of attention heads in clean run
$A_R\leftarrow M_\text{cache}(R)\qquad\qquad\qquad$ // Cache activations of attention heads in corrupted run
$A_{\nabla C}\leftarrow M_{\nabla\text{cache}}(C, F)\quad$ // Cache gradients of the metric w.r.t to attention heads in clean run
$\forall a\in M:P_a=\left|(A_{C,a}-A_{R,a})A_{\nabla C,a}\right|<\tau$
$M_\text{pruned}\leftarrow M$
**for** $a\in M$ **do**$\qquad\qquad\qquad\qquad\quad$// Loop over all activations in the model
$\qquad$ **if** $P_a$ **then**
$\qquad\qquad$ $M_{\text{pruned}, W_Q,a}\leftarrow0$ $\qquad$ // Prune by setting the query weight matrix of the node to 0
$\qquad$**end**
**end**
**return** $(M_\text{pruned},P)$
___

## Future work
### Method: Path attribution patching 
<!-- (First order approximation of Path Patching; already mostly finished) -->

For a specific pair of `sender` and `receiver` heads, we approximate their attribution to the metric as follows:

1. Caching activations and gradients:
**Two fwd passes:** caching activations for clean and corrupted prompt, yielding `clean_cache`, `corrupted_cache`
**One bwd pass**: caching gradients of the loss metric w.r.t. activations for the clean prompt, yielding `clean_grad_cache`
2. Approximate the effect of an edge on the logit difference using the formula `((clean_early_node_out - corrupted_early_node_out) * corrupted_late_node_in_grad).sum()`
Where `corrupted_late_node_in_grad` is the derivative of the metric with respect to the residual stream input immediately before the layer norm before the receiver node. 

<!-- We have already mostly finished this algorithm, but still need to fix certain parts. -->
### Follow Up:
Investigate the validity of the approximation when freezing vs not freezing ln in the receiver node when computing gradient *w.r.t. the input* of the receiver node.

### Experiments
- Test both **Node attribution patching** and **Path attribution patching** for identifying induction heads in a short transformer (currently debugging error regarding loss metric)

---
Code available at: https://github.com/rusheb/arena-hackathon-attribution-patching






    

    

<!-- ## Method

- summary of attribution patching
    
- summary of acdc
     -->

<!-- ** -->