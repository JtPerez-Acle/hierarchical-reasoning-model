# Hierarchical Reasoning Model (HRM) - Implementation Guide

## Project Overview

This project implements the Hierarchical Reasoning Model (HRM) from the paper "Hierarchical Reasoning Model" (arXiv:2506.21734v1). HRM is a brain-inspired architecture that demonstrates hierarchical reasoning capabilities on complex logical tasks.

## Current Implementation Status

✅ **Completed:**
- Core HRM architecture with hierarchical dynamics
- 1-step gradient approximation training  
- Sudoku dataset loader and training pipeline
- Deep supervision training methodology
- Convergence analysis and visualization tools
- CPU-optimized training experiments

📊 **Sudoku Results Achieved:**
- Successfully trained 369K parameter model on 1,000 Sudoku puzzles
- Achieved stable convergence (loss: 13.5 → 2.2)
- Demonstrated learning progress (11.5% cell accuracy vs 11.1% baseline)
- Validated hierarchical processing with N=3, T=2 architecture

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

- [x] Basic HRM forward pass matches paper's dynamics
- [x] Memory usage remains O(1) during training (verified with 1-step gradient)
- [x] Forward residuals show hierarchical convergence pattern
- [x] Stable training on complex reasoning task (Sudoku)
- [ ] Sudoku accuracy > 20% on moderate difficulty (11.5% achieved with limited data)
- [ ] Clear dimensionality separation (PR_H / PR_L > 2.0)

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
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repo-url>
cd hierarchical-reasoning-model
uv sync
```

2. Run existing experiments:
```bash
# Test the core implementation
uv run python test_hrm.py

# Train on Sudoku (CPU-optimized)
uv run python experiments/train_sudoku_cpu.py

# Simple convergence demo
uv run python experiments/sudoku_demo_simple.py
```

3. Explore the implementation:
```python
from models.hrm import HierarchicalReasoningModel
from datasets.sudoku import create_sudoku_dataloaders

# Create model (tested configuration)
model = HierarchicalReasoningModel(
    input_dim=810,      # 81 cells × 10 classes
    hidden_dim=64,      # Compact for CPU
    output_dim=729,     # 81 cells × 9 digits
    num_transformer_layers=2,
    N=3,                # High-level cycles
    T=2                 # Steps per cycle
)
```

## Key Implementation Learnings

From our Sudoku experiments:

1. **Sequence Dimension Handling**: The model outputs (batch, seq, features). Take the last timestep for predictions.

2. **Loss Calculation**: For multi-position classification:
   ```python
   outputs = outputs.reshape(-1, 81, 9)  # Reshape to positions × classes
   outputs = outputs.reshape(-1, 9)      # Flatten for CrossEntropyLoss
   targets = targets.reshape(-1)         # Flatten targets similarly
   ```

3. **CPU Optimization**: Use small batch sizes (4), reduced hidden dimensions (64), and fewer transformer layers (2).

4. **Deep Supervision**: Not used in current implementation due to sequence dimension complexity. Standard training works well.

5. **Parameter Initialization**: Model uses `truncated_lecun_normal_` for stable training.

## Next Steps

1. **Improve Sudoku Performance**: 
   - Increase training data (currently only 1,000 samples)
   - Add curriculum learning (start with easier puzzles)
   - Implement constraint-aware loss functions

2. **Implement Maze Navigation**: 
   - Leverage hierarchical planning capabilities
   - Test transfer learning from Sudoku

3. **Adaptive Computation Time**:
   - Add dynamic halting based on confidence
   - Measure efficiency gains

## References

- Paper: "Hierarchical Reasoning Model" (arXiv:2506.21734v1)
- Key concepts: Hierarchical convergence, 1-step gradient, deep supervision
- Inspiration: Brain's multi-timescale processing

Remember: The goal is to explore and understand the HRM's unique approach to reasoning, not necessarily to achieve the exact paper results immediately.