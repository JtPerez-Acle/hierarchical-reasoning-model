import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.hrm import HierarchicalReasoningModel
from datasets.sudoku import SudokuDataset, SudokuConstraintChecker


def simple_sudoku_demo():
    """Simple demonstration of HRM on Sudoku puzzles."""
    print("Loading Sudoku dataset...")
    dataset = SudokuDataset('datasets/sudoku.csv', max_samples=100, split='test')
    
    print("Creating HRM model...")
    model = HierarchicalReasoningModel(
        input_dim=810,  # 81 positions × 10 classes
        hidden_dim=128,
        output_dim=81 * 9,  # 81 positions × 9 digits
        num_transformer_layers=2,
        N=4,  # Number of high-level cycles
        T=3,  # Steps per cycle
        num_heads=4,
        dropout=0.1
    )
    
    # Get a puzzle
    puzzle_tensor, solution_tensor, info = dataset[0]
    puzzle_grid = SudokuDataset.decode_puzzle(puzzle_tensor, 'one_hot')
    solution_grid = SudokuDataset.decode_solution(solution_tensor)
    
    print(f"\nSelected puzzle with {info['empty_cells']} empty cells")
    print("\nInput puzzle:")
    print(puzzle_grid)
    print("\nTrue solution:")
    print(solution_grid)
    
    # Run model
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        input_batch = puzzle_tensor.unsqueeze(0)
        
        # Forward pass
        states, output = model(input_batch)
        
        # Get predictions
        output_reshaped = output.reshape(-1, 81, 9)
        predictions = output_reshaped.argmax(dim=-1)
        pred_grid = SudokuDataset.decode_solution(predictions[0])
        
        print("\nModel prediction (untrained):")
        print(pred_grid)
        
        # Check validity
        violations = SudokuConstraintChecker.count_violations(pred_grid)
        correct_cells = (pred_grid == solution_grid).sum()
        
        print(f"\nCorrect cells: {correct_cells}/81")
        print(f"Constraint violations: {violations}")
    
    # Visualize state evolution
    print("\nRunning step-by-step analysis...")
    
    # Track states through multiple steps
    batch_size = 1
    device = 'cpu'
    z_h, z_l = model.initialize_hidden_states(batch_size, device)
    input_embedding = model.input_embedding(input_batch.unsqueeze(1))
    
    l_norms = []
    h_norms = []
    predictions_over_time = []
    
    total_steps = model.N * model.T
    
    with torch.no_grad():
        for step in range(total_steps):
            # Update low-level
            z_l = model.low_level_step(z_l, z_h, input_embedding)
            
            # Update high-level at intervals
            if step % model.T == 0:
                z_h = model.high_level_step(z_h, z_l)
            
            # Track norms
            l_norms.append(torch.norm(z_l, dim=-1).mean().item())
            h_norms.append(torch.norm(z_h, dim=-1).mean().item())
            
            # Get current prediction
            output = model.output_head(z_h)
            output = output.squeeze(1).reshape(-1, 81, 9)
            pred = output.argmax(dim=-1)
            predictions_over_time.append(pred[0].cpu().numpy())
    
    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # State norms
    ax1.plot(l_norms, 'b-', label='Low-level norm', linewidth=2)
    ax1.plot(h_norms, 'r-', label='High-level norm', linewidth=2)
    
    # Mark H-module updates
    for i in range(0, total_steps, model.T):
        ax1.axvline(i, color='red', alpha=0.3, linestyle='--')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('State Norm')
    ax1.set_title('State Evolution During Reasoning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy over time
    accuracies = []
    for pred in predictions_over_time:
        pred_grid = pred.reshape(9, 9) + 1
        acc = (pred_grid == solution_grid).sum() / 81
        accuracies.append(acc)
    
    ax2.plot(accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cell Accuracy')
    ax2.set_title('Prediction Accuracy Over Steps')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/sudoku_convergence_simple.png')
    plt.show()
    
    print(f"\nFinal accuracy: {accuracies[-1]:.2%}")
    print("Note: Model is untrained, so random predictions are expected!")


if __name__ == "__main__":
    simple_sudoku_demo()