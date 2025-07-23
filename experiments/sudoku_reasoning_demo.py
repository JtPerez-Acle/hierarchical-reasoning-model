import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import seaborn as sns

from models.hrm import HierarchicalReasoningModel
from datasets.sudoku import SudokuDataset, SudokuConstraintChecker
from analysis.convergence import track_convergence


class SudokuReasoningVisualizer:
    """Visualize HRM's step-by-step reasoning process on Sudoku puzzles."""
    
    def __init__(self, model: HierarchicalReasoningModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
    
    def _track_solving_convergence(self, puzzle_tensor: torch.Tensor, max_steps: int) -> dict:
        """Track convergence metrics during solving."""
        self.model.eval()
        
        # Initialize
        batch_size = 1 if puzzle_tensor.dim() == 1 else puzzle_tensor.shape[0]
        if puzzle_tensor.dim() == 1:
            puzzle_tensor = puzzle_tensor.unsqueeze(0)
        
        z_h, z_l = self.model.initialize_hidden_states(batch_size, self.device)
        input_embedding = self.model.input_embedding(puzzle_tensor.unsqueeze(1))
        
        # Track metrics
        steps = []
        l_norms = []
        h_norms = []
        l_changes = [0]
        h_changes = [0]
        h_updates = []
        
        with torch.no_grad():
            for step in range(max_steps):
                # Store previous states
                prev_l = z_l.clone()
                prev_h = z_h.clone()
                
                # Update states
                z_l = self.model.low_level_step(z_l, z_h, input_embedding)
                
                if step % self.model.T == 0:
                    z_h = self.model.high_level_step(z_h, z_l)
                    h_updates.append(step)
                
                # Track norms and changes
                steps.append(step)
                l_norms.append(torch.norm(z_l, dim=-1).mean().item())
                h_norms.append(torch.norm(z_h, dim=-1).mean().item())
                
                if step > 0:
                    l_changes.append(torch.norm(z_l - prev_l, dim=-1).mean().item())
                    h_changes.append(torch.norm(z_h - prev_h, dim=-1).mean().item())
        
        return {
            'steps': steps,
            'l_norms': l_norms,
            'h_norms': h_norms,
            'l_changes': l_changes,
            'h_changes': h_changes,
            'h_updates': h_updates
        }
        
    def solve_with_steps(self, puzzle_tensor: torch.Tensor, max_steps: int = 20) -> dict:
        """
        Solve Sudoku puzzle and track intermediate steps.
        
        Returns:
            Dictionary containing predictions at each step and convergence data
        """
        self.model.eval()
        puzzle_tensor = puzzle_tensor.to(self.device)
        
        # Initialize hidden states
        batch_size = 1 if puzzle_tensor.dim() == 1 else puzzle_tensor.shape[0]
        if puzzle_tensor.dim() == 1:
            puzzle_tensor = puzzle_tensor.unsqueeze(0)
        
        z_h, z_l = self.model.initialize_hidden_states(batch_size, self.device)
        
        # Embed input
        input_embedding = self.model.input_embedding(puzzle_tensor.unsqueeze(1))  # Add sequence dimension
        
        # Track predictions at each step
        step_predictions = []
        step_confidences = []
        h_update_steps = []
        
        with torch.no_grad():
            for step in range(max_steps):
                # Update low-level state
                z_l = self.model.low_level_step(z_l, z_h, input_embedding)
                
                # Update high-level state at intervals
                if step % self.model.T == 0:
                    z_h = self.model.high_level_step(z_h, z_l)
                    h_update_steps.append(step)
                
                # Get current prediction from high-level state
                output = self.model.output_head(z_h)
                
                # Reshape to (batch, 81, 9) and get predictions
                output = output.squeeze(1)  # Remove sequence dimension
                output_reshaped = output.reshape(-1, 81, 9)
                probs = torch.softmax(output_reshaped, dim=-1)
                predictions = probs.argmax(dim=-1)
                confidences = probs.max(dim=-1).values
                
                step_predictions.append(predictions[0].cpu().numpy())  # Take first batch item
                step_confidences.append(confidences[0].cpu().numpy())
        
        # Also track convergence (using a different approach since track_convergence expects different params)
        convergence_data = self._track_solving_convergence(puzzle_tensor, max_steps)
        
        return {
            'predictions': step_predictions,
            'confidences': step_confidences,
            'h_updates': h_update_steps,
            'convergence': convergence_data
        }
    
    def visualize_solving_process(self, puzzle: np.ndarray, solution: np.ndarray, 
                                  solving_data: dict, save_path: str = None):
        """Create comprehensive visualization of the solving process."""
        steps = len(solving_data['predictions'])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main Sudoku grids
        ax_puzzle = fig.add_subplot(gs[0, 0])
        ax_current = fig.add_subplot(gs[0, 1])
        ax_confidence = fig.add_subplot(gs[0, 2])
        ax_solution = fig.add_subplot(gs[0, 3])
        
        # Convergence plots
        ax_convergence = fig.add_subplot(gs[1, :])
        
        # Metrics
        ax_accuracy = fig.add_subplot(gs[2, 0:2])
        ax_violations = fig.add_subplot(gs[2, 2:])
        
        # Plot initial puzzle
        self._plot_sudoku(ax_puzzle, puzzle, "Input Puzzle")
        self._plot_sudoku(ax_solution, solution, "True Solution")
        
        # Track metrics over steps
        accuracies = []
        violations = []
        empty_cells_filled = []
        
        # Animation function
        def animate(step):
            # Clear dynamic axes
            ax_current.clear()
            ax_confidence.clear()
            
            # Get current prediction
            prediction = solving_data['predictions'][step] + 1  # Convert to 1-9
            confidence = solving_data['confidences'][step]
            
            # Calculate current grid
            current_grid = puzzle.copy()
            empty_mask = (puzzle == 0)
            current_grid[empty_mask] = prediction.reshape(9, 9)[empty_mask]
            
            # Plot current state
            self._plot_sudoku(ax_current, current_grid, f"Step {step + 1}")
            
            # Highlight H-module updates
            if step in solving_data['h_updates']:
                ax_current.text(4, -1, "H-module updated!", ha='center', 
                              fontsize=12, color='red', weight='bold')
            
            # Plot confidence heatmap
            conf_grid = confidence.reshape(9, 9)
            sns.heatmap(conf_grid, ax=ax_confidence, cmap='YlOrRd', 
                       vmin=0, vmax=1, cbar_kws={'label': 'Confidence'})
            ax_confidence.set_title(f"Prediction Confidence")
            ax_confidence.invert_yaxis()
            
            # Calculate metrics
            correct = (current_grid == solution).sum()
            accuracy = correct / 81
            accuracies.append(accuracy)
            
            violation_count = SudokuConstraintChecker.count_violations(current_grid)
            violations.append(violation_count)
            
            filled = (current_grid[empty_mask] > 0).sum()
            empty_cells_filled.append(filled)
            
            # Update metric plots
            ax_accuracy.clear()
            ax_accuracy.plot(accuracies, 'b-', linewidth=2)
            ax_accuracy.set_xlabel('Step')
            ax_accuracy.set_ylabel('Cell Accuracy')
            ax_accuracy.set_title('Accuracy Progress')
            ax_accuracy.set_ylim(0, 1)
            ax_accuracy.grid(True, alpha=0.3)
            
            ax_violations.clear()
            ax_violations.plot(violations, 'r-', linewidth=2)
            ax_violations.set_xlabel('Step')
            ax_violations.set_ylabel('Constraint Violations')
            ax_violations.set_title('Sudoku Rule Violations')
            ax_violations.grid(True, alpha=0.3)
            
            # Update convergence plot
            self._plot_convergence_subplot(ax_convergence, solving_data['convergence'], step)
            
            return []
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=steps, interval=500, repeat=True)
        
        if save_path:
            # Save as gif
            anim.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved to {save_path}")
        
        plt.tight_layout()
        return fig, anim
    
    def _plot_sudoku(self, ax, grid, title):
        """Plot a Sudoku grid with proper formatting."""
        ax.clear()
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            ax.axhline(i - 0.5, color='black', linewidth=lw)
            ax.axvline(i - 0.5, color='black', linewidth=lw)
        
        # Fill in numbers
        for row in range(9):
            for col in range(9):
                value = int(grid[row, col])
                if value > 0:
                    # Original numbers in black, predicted in blue
                    color = 'black' if hasattr(self, '_original_puzzle') and \
                            self._original_puzzle[row, col] > 0 else 'blue'
                    ax.text(col, 8-row, str(value), 
                           ha='center', va='center', 
                           fontsize=14, color=color, weight='bold')
    
    def _plot_convergence_subplot(self, ax, convergence_data, current_step):
        """Plot convergence data up to current step."""
        ax.clear()
        
        steps = convergence_data['steps'][:current_step+1]
        l_norms = convergence_data['l_norms'][:current_step+1]
        h_norms = convergence_data['h_norms'][:current_step+1]
        l_changes = convergence_data['l_changes'][:current_step+1]
        h_changes = convergence_data['h_changes'][:current_step+1]
        
        # Plot state norms
        ax2 = ax.twinx()
        line1 = ax.plot(steps, l_norms, 'b-', label='L-module norm', linewidth=2)
        line2 = ax.plot(steps, h_norms, 'r-', label='H-module norm', linewidth=2)
        
        # Plot changes
        if len(steps) > 1:
            line3 = ax2.plot(steps[1:], l_changes[1:current_step+1], 'b--', alpha=0.5, label='L-module change')
            line4 = ax2.plot(steps[1:], h_changes[1:current_step+1], 'r--', alpha=0.5, label='H-module change')
        else:
            line3 = []
            line4 = []
        
        # Mark H-module updates
        h_updates = convergence_data['h_updates']
        for h_step in h_updates:
            if h_step <= current_step:
                ax.axvline(h_step, color='red', alpha=0.3, linestyle=':')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('State Norm', color='black')
        ax2.set_ylabel('State Change', color='gray')
        ax.set_title('Module Convergence Dynamics')
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def create_difficulty_analysis(self, dataset: SudokuDataset, num_samples: int = 20):
        """Analyze how solving performance varies with puzzle difficulty."""
        difficulties = []
        solve_steps = []
        final_accuracies = []
        
        # Sample puzzles across difficulty range
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for idx in indices:
            puzzle_tensor, solution_tensor, info = dataset[idx]
            difficulty = info['empty_cells']
            
            # Solve puzzle
            solving_data = self.solve_with_steps(puzzle_tensor.unsqueeze(0))
            
            # Find convergence point (when accuracy stops improving significantly)
            predictions = solving_data['predictions']
            solution_grid = SudokuDataset.decode_solution(solution_tensor)
            
            accuracies = []
            for pred in predictions:
                pred_grid = pred.reshape(9, 9) + 1
                acc = (pred_grid == solution_grid).sum() / 81
                accuracies.append(acc)
            
            # Find convergence step
            converged_step = len(accuracies) - 1
            for i in range(len(accuracies) - 5):
                if abs(accuracies[i+5] - accuracies[i]) < 0.01:
                    converged_step = i + 5
                    break
            
            difficulties.append(difficulty)
            solve_steps.append(converged_step)
            final_accuracies.append(accuracies[-1])
        
        # Create scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Steps vs difficulty
        ax1.scatter(difficulties, solve_steps, alpha=0.6)
        ax1.set_xlabel('Number of Empty Cells')
        ax1.set_ylabel('Steps to Convergence')
        ax1.set_title('Solving Complexity vs Puzzle Difficulty')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy vs difficulty
        ax2.scatter(difficulties, final_accuracies, alpha=0.6)
        ax2.set_xlabel('Number of Empty Cells')
        ax2.set_ylabel('Final Accuracy')
        ax2.set_title('Solving Accuracy vs Puzzle Difficulty')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def demo_sudoku_reasoning():
    """Demonstrate HRM solving Sudoku puzzles with visualization."""
    # Load a small subset of data
    dataset = SudokuDataset('datasets/sudoku.csv', max_samples=100, split='test')
    
    # Create a simple HRM model
    model = HierarchicalReasoningModel(
        input_dim=810,  # 81 positions × 10 classes
        hidden_dim=128,
        output_dim=81 * 9,  # 81 positions × 9 digits
        num_transformer_layers=2,
        N=6,  # Number of high-level cycles
        T=4,  # Steps per cycle
        num_heads=4,
        dropout=0.1
    )
    
    # Initialize visualizer
    visualizer = SudokuReasoningVisualizer(model)
    
    # Pick an interesting puzzle (medium difficulty)
    target_difficulty = 40  # Around 40 empty cells
    selected_idx = None
    for i in range(len(dataset)):
        _, _, info = dataset[i]
        if abs(info['empty_cells'] - target_difficulty) < 5:
            selected_idx = i
            break
    
    if selected_idx is None:
        selected_idx = 0
    
    # Get puzzle data
    puzzle_tensor, solution_tensor, info = dataset[selected_idx]
    puzzle_grid = SudokuDataset.decode_puzzle(puzzle_tensor, 'one_hot')
    solution_grid = SudokuDataset.decode_solution(solution_tensor)
    
    print(f"Selected puzzle with {info['empty_cells']} empty cells")
    print("\nInput puzzle:")
    print(puzzle_grid)
    
    # Solve with tracking
    solving_data = visualizer.solve_with_steps(puzzle_tensor.unsqueeze(0), max_steps=24)  # N*T = 6*4
    
    # Store original puzzle for visualization
    visualizer._original_puzzle = puzzle_grid
    
    # Create visualization
    fig, anim = visualizer.visualize_solving_process(
        puzzle_grid, solution_grid, solving_data,
        save_path='experiments/sudoku_solving_demo.gif'
    )
    
    plt.show()
    
    # Analyze difficulty scaling
    print("\nAnalyzing difficulty scaling...")
    diff_fig = visualizer.create_difficulty_analysis(dataset, num_samples=20)
    diff_fig.savefig('experiments/sudoku_difficulty_analysis.png')
    plt.show()


if __name__ == "__main__":
    demo_sudoku_reasoning()