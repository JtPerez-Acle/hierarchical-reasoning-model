import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json

from models.hrm import HierarchicalReasoningModel
from datasets.sudoku import create_sudoku_dataloaders, SudokuDataset, SudokuConstraintChecker
from training.gradient import DeepSupervisionTrainer
from analysis.visualization import HRMVisualizer
from analysis.convergence import track_convergence, plot_convergence_patterns


class SudokuHRMExperiment:
    """Experiment class for training HRM on Sudoku solving."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"experiments/sudoku_results_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(f"{self.output_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Initialize model
        self.model = self._create_model()
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_sudoku_dataloaders(
            config['data_path'],
            batch_size=config['batch_size'],
            max_samples=config.get('max_samples'),
            representation=config['representation'],
            num_workers=config.get('num_workers', 0)
        )
        
        # Initialize trainer
        self.trainer = DeepSupervisionTrainer(
            model=self.model,
            optimizer=optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0.01)
            ),
            num_segments=config['num_segments']
        )
        
        # Initialize visualizer
        self.visualizer = HRMVisualizer(self.model)
        
        # Metrics storage
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_cell_acc': [],
            'val_puzzle_acc': [],
            'val_violations': []
        }
    
    def _create_model(self) -> HierarchicalReasoningModel:
        """Create HRM model with appropriate dimensions for Sudoku."""
        config = self.config
        
        if config['representation'] == 'one_hot':
            input_dim = 810  # 81 positions × 10 classes
        else:
            input_dim = 81  # Direct position indices
        
        output_dim = 81 * 9  # 81 positions × 9 classes (digits 1-9)
        
        model = HierarchicalReasoningModel(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=output_dim,
            num_transformer_layers=config['num_transformer_layers'],
            N=config['N'],
            T=config['T'],
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
        
        return model.to(self.device)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (puzzles, solutions, info) in enumerate(progress_bar):
            puzzles = puzzles.to(self.device)
            solutions = solutions.to(self.device)
            
            # Deep supervision training step
            loss = self.trainer.train_step(puzzles, solutions)
            total_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss:.4f}"})
            
            # Periodic visualization
            if batch_idx % self.config.get('vis_freq', 100) == 0 and batch_idx > 0:
                self._visualize_training_state(epoch, batch_idx)
        
        return total_loss / len(self.train_loader)
    
    def validate(self, loader: DataLoader, phase: str = 'val'):
        """Validate model performance."""
        self.model.eval()
        total_loss = 0
        total_cell_correct = 0
        total_puzzle_correct = 0
        total_violations = 0
        total_cells = 0
        total_puzzles = 0
        
        with torch.no_grad():
            for puzzles, solutions, info in tqdm(loader, desc=f"Validating {phase}"):
                puzzles = puzzles.to(self.device)
                solutions = solutions.to(self.device)
                
                # Get model predictions
                loss, outputs = self.trainer.validate_step(puzzles, solutions)
                total_loss += loss
                
                # Reshape outputs to (batch, 81, 9)
                outputs = outputs.reshape(-1, 81, 9)
                predictions = outputs.argmax(dim=-1)  # (batch, 81)
                
                # Cell-level accuracy
                cell_correct = (predictions == solutions).sum().item()
                total_cell_correct += cell_correct
                total_cells += solutions.numel()
                
                # Puzzle-level accuracy
                puzzle_correct = (predictions == solutions).all(dim=1).sum().item()
                total_puzzle_correct += puzzle_correct
                total_puzzles += len(solutions)
                
                # Check constraint violations
                for i in range(len(predictions)):
                    pred_grid = SudokuDataset.decode_solution(predictions[i])
                    violations = SudokuConstraintChecker.count_violations(pred_grid)
                    total_violations += violations
        
        metrics = {
            'loss': total_loss / len(loader),
            'cell_acc': total_cell_correct / total_cells,
            'puzzle_acc': total_puzzle_correct / total_puzzles,
            'avg_violations': total_violations / total_puzzles
        }
        
        return metrics
    
    def _visualize_training_state(self, epoch: int, batch_idx: int):
        """Create visualization of current training state."""
        # Get a sample batch for visualization
        sample_batch = next(iter(self.val_loader))
        puzzles, solutions, info = sample_batch
        puzzles = puzzles[:4].to(self.device)  # Visualize 4 examples
        solutions = solutions[:4].to(self.device)
        
        # Track convergence
        with torch.no_grad():
            convergence_data = track_convergence(
                self.model, puzzles, max_steps=self.config['N'] * self.config['T']
            )
        
        # Create convergence plot
        fig = plot_convergence_patterns(convergence_data)
        fig.savefig(f"{self.output_dir}/convergence_epoch{epoch}_batch{batch_idx}.png")
        plt.close(fig)
        
        # Visualize predictions
        self._visualize_predictions(puzzles, solutions, epoch, batch_idx)
    
    def _visualize_predictions(self, puzzles: torch.Tensor, solutions: torch.Tensor, 
                               epoch: int, batch_idx: int):
        """Visualize model predictions on sample puzzles."""
        self.model.eval()
        
        with torch.no_grad():
            # Get predictions through multiple steps
            outputs = self.model(puzzles)
            outputs = outputs.reshape(-1, 81, 9)
            predictions = outputs.argmax(dim=-1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(puzzles), 3, figsize=(12, 4*len(puzzles)))
        if len(puzzles) == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(len(puzzles)):
            # Decode tensors
            puzzle_grid = SudokuDataset.decode_puzzle(puzzles[i], self.config['representation'])
            solution_grid = SudokuDataset.decode_solution(solutions[i])
            pred_grid = SudokuDataset.decode_solution(predictions[i])
            
            # Plot puzzle
            self._plot_sudoku(axes[i, 0], puzzle_grid, "Input Puzzle")
            
            # Plot solution
            self._plot_sudoku(axes[i, 1], solution_grid, "True Solution")
            
            # Plot prediction
            self._plot_sudoku(axes[i, 2], pred_grid, "Model Prediction")
            
            # Highlight errors
            errors = (pred_grid != solution_grid) & (solution_grid > 0)
            if errors.any():
                for row in range(9):
                    for col in range(9):
                        if errors[row, col]:
                            axes[i, 2].add_patch(plt.Rectangle(
                                (col-0.5, 8.5-row), 1, 1, 
                                fill=False, edgecolor='red', linewidth=2
                            ))
        
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/predictions_epoch{epoch}_batch{batch_idx}.png")
        plt.close(fig)
    
    def _plot_sudoku(self, ax, grid, title):
        """Plot a single Sudoku grid."""
        ax.set_title(title)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        
        # Draw thick lines for 3x3 boxes
        for i in range(0, 10, 3):
            ax.axhline(i - 0.5, color='black', linewidth=2)
            ax.axvline(i - 0.5, color='black', linewidth=2)
        
        # Fill in numbers
        for row in range(9):
            for col in range(9):
                value = grid[row, col]
                if value > 0:
                    ax.text(col, 8-row, str(value), 
                           ha='center', va='center', fontsize=12)
    
    def run_experiment(self):
        """Run the full training experiment."""
        print(f"Starting Sudoku HRM experiment on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            self.metrics_history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate(self.val_loader, 'val')
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['val_cell_acc'].append(val_metrics['cell_acc'])
            self.metrics_history['val_puzzle_acc'].append(val_metrics['puzzle_acc'])
            self.metrics_history['val_violations'].append(val_metrics['avg_violations'])
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Cell Acc: {val_metrics['cell_acc']:.3f}, "
                  f"Puzzle Acc: {val_metrics['puzzle_acc']:.3f}, "
                  f"Avg Violations: {val_metrics['avg_violations']:.2f}")
            
            # Save best model
            if val_metrics['cell_acc'] > best_val_acc:
                best_val_acc = val_metrics['cell_acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config
                }, f"{self.output_dir}/best_model.pt")
            
            # Periodic comprehensive visualization
            if epoch % self.config.get('vis_epoch_freq', 5) == 0:
                self._create_comprehensive_visualization(epoch)
        
        # Final test evaluation
        print("\nFinal test evaluation:")
        test_metrics = self.validate(self.test_loader, 'test')
        print(f"Test Cell Acc: {test_metrics['cell_acc']:.3f}, "
              f"Test Puzzle Acc: {test_metrics['puzzle_acc']:.3f}, "
              f"Test Avg Violations: {test_metrics['avg_violations']:.2f}")
        
        # Save final metrics
        self._save_final_results(test_metrics)
    
    def _create_comprehensive_visualization(self, epoch: int):
        """Create comprehensive visualization dashboard."""
        dashboard = self.visualizer.create_training_dashboard(
            self.metrics_history,
            title=f"Sudoku HRM Training - Epoch {epoch}"
        )
        dashboard.savefig(f"{self.output_dir}/dashboard_epoch{epoch}.png", dpi=150)
        plt.close(dashboard)
    
    def _save_final_results(self, test_metrics: dict):
        """Save final experiment results."""
        results = {
            'config': self.config,
            'metrics_history': self.metrics_history,
            'test_metrics': test_metrics,
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
        
        with open(f"{self.output_dir}/final_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        
        # Cell accuracy
        axes[0, 1].plot(self.metrics_history['val_cell_acc'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cell Accuracy')
        axes[0, 1].set_title('Cell-Level Accuracy')
        
        # Puzzle accuracy
        axes[1, 0].plot(self.metrics_history['val_puzzle_acc'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Puzzle Accuracy')
        axes[1, 0].set_title('Full Puzzle Accuracy')
        
        # Constraint violations
        axes[1, 1].plot(self.metrics_history['val_violations'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Average Violations')
        axes[1, 1].set_title('Constraint Violations')
        
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/training_curves.png")
        plt.close(fig)


def main():
    """Run Sudoku HRM experiment with default configuration."""
    config = {
        # Data
        'data_path': 'datasets/sudoku.csv',
        'batch_size': 32,
        'max_samples': 10000,  # Use subset for faster experimentation
        'representation': 'one_hot',
        
        # Model
        'hidden_dim': 256,
        'num_transformer_layers': 3,
        'N': 5,  # Number of high-level cycles
        'T': 4,  # High-level update frequency
        'num_heads': 8,
        'dropout': 0.1,
        
        # Training
        'num_epochs': 50,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'num_segments': 4,  # For deep supervision
        
        # Visualization
        'vis_freq': 100,  # Batch frequency
        'vis_epoch_freq': 5,  # Epoch frequency
        'num_workers': 0
    }
    
    experiment = SudokuHRMExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()