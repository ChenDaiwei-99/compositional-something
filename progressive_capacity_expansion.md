# Progressive Capacity Expansion for Self-Improving Models

## Overview

In a self-improvement pipeline, model capacity may become a bottleneck as training progresses.  
Instead of restarting training with a larger model, we can expand the existing model while preserving its learned function.

Two mechanisms:

- Net2Net expansion (general neural networks)
- Residual expansion for transformers

### The pipeline

```
model_t
↓
train on dataset_t
↓
detect capacity saturation
↓
expand model capacity
↓
continue training
```

---

## Method 1: Net2Net Expansion

Reference:  
**Net2Net: Accelerating Learning via Knowledge Transfer**

Net2Net introduces **function-preserving transformations** that allow increasing model capacity without changing the model's output.

---

### Net2WiderNet (Width Expansion)

**Goal:** increase hidden dimension.

Example:

```
hidden_dim: 512 → 1024
```

Procedure:

1. Select neurons to replicate.
2. Copy their weights.
3. Adjust downstream weights to keep outputs unchanged.

Conceptually:

```
original neuron
↓
duplicate neuron
↓
split outgoing weights
```

Result:

```
f_new(x) = f_old(x)
```

Training can continue immediately.

---

### Net2DeeperNet (Depth Expansion)

**Goal:** add new layers.

Example:

```
Layer1
Layer2
Layer3
```

becomes

```
Layer1
Layer2
IdentityLayer
Layer3
```

Initialization:

```
IdentityLayer(x) = x
```

So

```
f_new(x) = f_old(x)
```

After expansion, training continues normally.

---

## Method 2: Transformer Residual Expansion

Standard transformer block:

```
x_{l+1} = x_l + F(x_l)
```

where `F` contains:

- attention
- MLP
- normalization

---

### Problem

Naively duplicating layers changes the function:

```
x_{l+2} = x_l + F(x_l) + F(x_l + F(x_l))
```

which is **not equal** to the original mapping.

---

### Residual Scaling Solution

Split the residual update across duplicated layers.

Original block:

```
x = x + F(x)
```

Expanded blocks:

```
x = x + (1/k) * F(x)
x = x + (1/k) * F(x)
...
k times
```

Example for doubling depth:

```
x = x + 0.5 * F1(x)
x = x + 0.5 * F2(x)
```

Initialization:

```
F1 = copy of original block
F2 = copy of original block
```

This approximates the same function while increasing depth.

---

### Interpretation

Transformer ≈ discretized ODE.

Increasing depth corresponds to **smaller integration step size**.
