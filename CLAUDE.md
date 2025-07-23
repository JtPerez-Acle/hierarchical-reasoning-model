# Hierarchical Reasoning Model (HRM) Exploration Guide

## Project Overview

This project implements the Hierarchical Reasoning Model (HRM) from the paper "Hierarchical Reasoning Model" (arXiv:2506.21734v1). HRM is a brain-inspired architecture that achieves state-of-the-art reasoning performance with only 27M parameters and 1000 training examples.

## Key Objectives

1. Implement the core HRM architecture with hierarchical convergence
2. Reproduce the 1-step gradient approximation training
3. Test on simplified versions of ARC-AGI, Sudoku, and Maze tasks
4. Visualize the hierarchical dynamics and dimensionality separation
5. Experiment with Adaptive Computation Time (ACT)

## Development Environment

- Python 3.11
- PyTorch 2.0+
- NumPy, Matplotlib for visualization
- Optional: einops for tensor operations

## Phase 1: Core Architecture Implementation

### Task 1.1: Basic Components
Create the fundamental building blocks:

```python
# File: models/components.py
- RMSNorm implementation
- Rotary Positional Encoding (RoPE)
- Gated Linear Unit (GLU)
- TransformerBlock with above components
```

### Task 1.2: HRM Module Structure
Implement the main HRM class:

```python
# File: models/hrm.py
class HRM:
    - Input embedding layer
    - Low-level module (L_net)
    - High-level module (H_net)
    - Output head
    - Forward pass with hierarchical dynamics
```

Key implementation details:
- L-module updates every step
- H-module updates every T steps
- Use torch.no_grad() for all but final step

### Task 1.3: One-Step Gradient Implementation
Implement the efficient gradient approximation:

```python
# File: training/gradient.py
- Detach intermediate states
- Only backprop through final states
- Verify O(1) memory usage
```

## Phase 2: Dataset Preparation

### Task 2.1: Sudoku Dataset
Create a simple Sudoku dataset:

```python
# File: datasets/sudoku.py
- Generate/load easy and hard Sudoku puzzles
- Implement difficulty measurement (backtrack counting)
- Create DataLoader with flattened grid representation
```

### Task 2.2: Simple Maze Dataset
Create maze navigation dataset:

```python
# File: datasets/maze.py
- Generate random mazes (start with 10x10, scale to 30x30)
- Compute optimal paths using BFS
- Create input/output pairs
```

### Task 2.3: Simplified ARC-AGI
Create a minimal ARC-like dataset:

```python
# File: datasets/arc_simple.py
- Start with pattern completion tasks
- Implement basic augmentations
- Create learnable task tokens
```

## Phase 3: Training Implementation

### Task 3.1: Deep Supervision Training
Implement the segmented training approach:

```python
# File: training/deep_supervision.py
def train_with_deep_supervision(model, data, segments):
    for x, y in data:
        z = initial_state
        for segment in range(segments):
            z, y_hat = model(z, x)
            loss = criterion(y_hat, y)
            z = z.detach()  # Critical!
            loss.backward()
            optimizer.step()
```

### Task 3.2: Basic Training Loop
Create standard training infrastructure:

```python
# File: training/train.py
- Training loop with logging
- Validation evaluation
- Checkpoint saving
- Metric tracking
```

## Phase 4: Experiments and Analysis

### Task 4.1: Convergence Analysis
Visualize hierarchical convergence:

```python
# File: analysis/convergence.py
- Track forward residuals at each step
- Plot L-module vs H-module convergence patterns
- Compare with standard RNN baseline
```

### Task 4.2: Dimensionality Analysis
Measure participation ratio:

```python
# File: analysis/dimensionality.py
- Collect hidden states during inference
- Compute covariance matrices
- Calculate participation ratio for L and H modules
- Verify dimensionality hierarchy
```

### Task 4.3: Visualization of Reasoning
Create intermediate step visualizations:

```python
# File: analysis/visualization.py
- Decode intermediate predictions
- Visualize Sudoku solving steps
- Show maze exploration process
- Track solution evolution
```

## Phase 5: Advanced Features

### Task 5.1: Adaptive Computation Time
Implement ACT with Q-learning:

```python
# File: models/act.py
- Q-head for halt/continue decisions
- Stochastic halting mechanism
- Joint training with main task
```

### Task 5.2: Inference-Time Scaling
Test computational scaling:

```python
# File: experiments/scaling.py
- Train with fixed Mmax
- Test with larger Mmax at inference
- Measure performance improvements
```

## Exploration Questions

1. **Depth vs Width**: How does performance scale with model depth (N*T) vs width (hidden_dim)?
2. **Minimal Working Example**: What's the smallest HRM that shows hierarchical convergence?
3. **Task Transfer**: Can an HRM trained on Sudoku help with maze solving?
4. **Gradient Approximation**: How does 1-step compare to 2-step or full BPTT?
5. **Biological Plausibility**: Can we add noise/sparsity while maintaining performance?

## Success Metrics

- [ ] Basic HRM forward pass matches paper's dynamics
- [ ] Sudoku accuracy > 20% on moderate difficulty
- [ ] Clear dimensionality separation (PR_H / PR_L > 2.0)
- [ ] Forward residuals show hierarchical convergence pattern
- [ ] Memory usage remains O(1) during training

## Implementation Tips

1. **Start Small**: Begin with 2x2 Sudoku or 5x5 mazes
2. **Debug Carefully**: Print shapes and check gradient flow
3. **Monitor Convergence**: Log forward residuals at each step
4. **Validate Assumptions**: Ensure H-module only updates every T steps
5. **Use Assertions**: Verify detachment prevents gradient flow

## Code Structure

```
hrm-exploration/
├── models/
│   ├── __init__.py
│   ├── components.py      # Basic building blocks
│   ├── hrm.py            # Main HRM model
│   └── act.py            # Adaptive computation
├── datasets/
│   ├── __init__.py
│   ├── sudoku.py         # Sudoku dataset
│   ├── maze.py           # Maze dataset
│   └── arc_simple.py     # Simplified ARC tasks
├── training/
│   ├── __init__.py
│   ├── train.py          # Training loops
│   ├── gradient.py       # 1-step approximation
│   └── deep_supervision.py
├── analysis/
│   ├── __init__.py
│   ├── convergence.py    # Convergence analysis
│   ├── dimensionality.py # PR calculations
│   └── visualization.py  # Step visualizations
├── experiments/
│   ├── __init__.py
│   └── scaling.py        # Scaling experiments
└── main.py               # Entry point
```

## Getting Started

1. Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch numpy matplotlib einops
```

2. Start with the simplest component:
```python
# Begin with RMSNorm implementation
# Then build up to TransformerBlock
# Finally assemble into HRM
```

3. Create a minimal test:
```python
# Test hierarchical updates
model = HRM(input_dim=10, hidden_dim=64, output_dim=10, N=2, T=3)
x = torch.randn(1, 10)
output = model(x)
# Verify shape and gradient flow
```

## Notes for Claude Code

- Focus on clarity over optimization initially
- Add extensive comments explaining the hierarchical dynamics
- Create unit tests for each component
- Use descriptive variable names (e.g., `high_level_state` not `zH`)
- Implement logging to track the convergence behavior
- Start with CPU-only implementation, add GPU support later

## References

- Paper: "Hierarchical Reasoning Model" (arXiv:2506.21734v1)
- Key concepts: Hierarchical convergence, 1-step gradient, deep supervision
- Inspiration: Brain's multi-timescale processing

Remember: The goal is to explore and understand the HRM's unique approach to reasoning, not necessarily to achieve the exact paper results immediately.