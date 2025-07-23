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


class CPUSudokuExperiment:
    """CPU-optimized Sudoku HRM experiment."""
    
    def __init__(self):
        # CPU-optimized configuration
        self.config = {
            # Data - small subset for CPU training
            'data_path': 'datasets/sudoku.csv',
            'batch_size': 4,  # Very small batches for CPU
            'max_samples': 1000,  # Small dataset
            'representation': 'one_hot',
            
            # Model - smaller architecture
            'hidden_dim': 64,  # Reduced from 256
            'num_transformer_layers': 2,  # Reduced from 3
            'N': 3,  # Fewer cycles
            'T': 2,  # Shorter cycles
            'num_heads': 4,  # Fewer heads
            'dropout': 0.1,
            
            # Training - fewer epochs, faster convergence
            'num_epochs': 10,
            'learning_rate': 1e-3,  # Higher LR for faster convergence
            'weight_decay': 0.01,
            'num_segments': 2,  # Fewer segments for deep supervision
            
            # Monitoring
            'print_freq': 10,  # Print every 10 batches
            'eval_freq': 2,    # Evaluate every 2 epochs
        }
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"experiments/sudoku_cpu_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(f"{self.output_dir}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Starting CPU-optimized Sudoku experiment")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize model
        self.model = self._create_model()
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_sudoku_dataloaders(
            self.config['data_path'],
            batch_size=self.config['batch_size'],
            max_samples=self.config['max_samples'],
            representation=self.config['representation'],
            num_workers=0  # No multiprocessing on CPU
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        
        # Initialize trainer and optimizer
        self.trainer = DeepSupervisionTrainer(
            model=self.model,
            segments_per_example=self.config['num_segments']
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_cell_acc': [],
            'val_puzzle_acc': [],
            'val_violations': []
        }
    
    def _create_model(self) -> HierarchicalReasoningModel:
        """Create CPU-optimized HRM model."""
        config = self.config
        
        input_dim = 810  # 81 positions × 10 classes
        output_dim = 81 * 9  # 81 positions × 9 classes (digits 1-9)
        
        model = HierarchicalReasoningModel(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=output_dim,
            num_transformer_layers=config['num_transformer_layers'],
            N=config['N'],
            T=config['T'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        
        return model
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with progress tracking."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']}"
        )
        
        for batch_idx, (puzzles, solutions, info) in enumerate(progress_bar):
            # Deep supervision training step - need custom loss handling
            self.optimizer.zero_grad()
            
            # Forward pass
            _, outputs = self.model(puzzles)
            
            # Handle sequence dimension - take the last timestep
            if outputs.dim() == 3:  # (batch, seq, features)
                outputs = outputs[:, -1, :]  # Take last sequence step: (batch, features)
            
            # Reshape for CrossEntropyLoss: (batch, 729) -> (batch, 81, 9) -> (batch*81, 9)
            outputs = outputs.reshape(-1, 81, 9)  # (batch, 81, 9)
            outputs = outputs.reshape(-1, 9)      # (batch*81, 9)
            solutions_flat = solutions.reshape(-1) # (batch*81,)
            
            # Compute loss
            loss = self.criterion(outputs, solutions_flat)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
            
            # Periodic logging
            if batch_idx % self.config['print_freq'] == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}: Loss = {loss:.4f}")
        
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
            for puzzles, solutions, info in tqdm(loader, desc=f"Evaluating {phase}"):
                # Get model predictions
                _, outputs = self.model(puzzles)
                
                # Handle sequence dimension - take the last timestep
                if outputs.dim() == 3:  # (batch, seq, features)
                    outputs = outputs[:, -1, :]  # Take last sequence step: (batch, features)
                
                # Reshape for loss calculation
                outputs_reshaped = outputs.reshape(-1, 81, 9)  # (batch, 81, 9)
                outputs_flat = outputs_reshaped.reshape(-1, 9)  # (batch*81, 9)
                solutions_flat = solutions.reshape(-1)          # (batch*81,)
                
                loss = self.criterion(outputs_flat, solutions_flat)
                total_loss += loss.item()
                
                # Get predictions
                predictions = outputs_reshaped.argmax(dim=-1)  # (batch, 81)
                
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
    
    def _save_sample_predictions(self, epoch: int):
        """Save sample predictions for visualization."""
        self.model.eval()
        
        # Get a few samples
        sample_batch = next(iter(self.val_loader))
        puzzles, solutions, info = sample_batch
        puzzles = puzzles[:2]  # Just 2 samples
        solutions = solutions[:2]
        
        with torch.no_grad():
            _, outputs = self.model(puzzles)
            
            # Handle sequence dimension
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]
            
            outputs = outputs.reshape(-1, 81, 9)
            predictions = outputs.argmax(dim=-1)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for i in range(2):
            # Decode grids
            puzzle_grid = SudokuDataset.decode_puzzle(puzzles[i], 'one_hot')
            solution_grid = SudokuDataset.decode_solution(solutions[i])
            pred_grid = SudokuDataset.decode_solution(predictions[i])
            
            # Plot grids
            self._plot_sudoku(axes[i, 0], puzzle_grid, f"Puzzle {i+1}")
            self._plot_sudoku(axes[i, 1], solution_grid, f"Solution {i+1}")
            self._plot_sudoku(axes[i, 2], pred_grid, f"Prediction {i+1}")
            
            # Highlight errors
            errors = pred_grid != solution_grid
            if errors.any():
                for row in range(9):
                    for col in range(9):
                        if errors[row, col]:
                            axes[i, 2].add_patch(plt.Rectangle(
                                (col-0.5, 8.5-row), 1, 1, 
                                fill=False, edgecolor='red', linewidth=2
                            ))
        
        plt.tight_layout()
        fig.suptitle(f'Sample Predictions - Epoch {epoch+1}', y=0.98)
        fig.savefig(f"{self.output_dir}/predictions_epoch_{epoch+1}.png", bbox_inches='tight')
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
                           ha='center', va='center', fontsize=10)
    
    def run_experiment(self):
        """Run the full training experiment."""
        print(f"\nStarting training...")
        best_val_acc = 0
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            self.metrics['train_loss'].append(train_loss)
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Validate periodically
            if epoch % self.config['eval_freq'] == 0 or epoch == self.config['num_epochs'] - 1:
                val_metrics = self.validate(self.val_loader, 'val')
                
                self.metrics['val_loss'].append(val_metrics['loss'])
                self.metrics['val_cell_acc'].append(val_metrics['cell_acc'])
                self.metrics['val_puzzle_acc'].append(val_metrics['puzzle_acc'])
                self.metrics['val_violations'].append(val_metrics['avg_violations'])
                
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Cell Acc: {val_metrics['cell_acc']:.3f}")
                print(f"  Puzzle Acc: {val_metrics['puzzle_acc']:.3f}")
                print(f"  Avg Violations: {val_metrics['avg_violations']:.2f}")
                
                # Save best model
                if val_metrics['cell_acc'] > best_val_acc:
                    best_val_acc = val_metrics['cell_acc']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'metrics': val_metrics,
                        'config': self.config
                    }, f"{self.output_dir}/best_model.pt")
                    print(f"  -> New best model saved!")
                
                # Save sample predictions
                self._save_sample_predictions(epoch)
        
        # Final test evaluation
        print(f"\nFinal test evaluation:")
        test_metrics = self.validate(self.test_loader, 'test')
        print(f"Test Cell Acc: {test_metrics['cell_acc']:.3f}")
        print(f"Test Puzzle Acc: {test_metrics['puzzle_acc']:.3f}")
        print(f"Test Avg Violations: {test_metrics['avg_violations']:.2f}")
        
        # Save final results and plots
        self._save_final_results(test_metrics)
        
        print(f"\nExperiment completed!")
        print(f"Results saved to: {self.output_dir}")
    
    def _save_final_results(self, test_metrics: dict):
        """Save final experiment results and plots."""
        results = {
            'config': self.config,
            'metrics': self.metrics,
            'test_metrics': test_metrics,
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
        
        with open(f"{self.output_dir}/final_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        epochs = list(range(len(self.metrics['train_loss'])))
        val_epochs = list(range(0, len(self.metrics['train_loss']), self.config['eval_freq']))
        if len(self.metrics['train_loss']) - 1 not in val_epochs:
            val_epochs.append(len(self.metrics['train_loss']) - 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(val_epochs, self.metrics['val_loss'], 'r-', label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cell accuracy
        axes[0, 1].plot(val_epochs, self.metrics['val_cell_acc'], 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cell Accuracy')
        axes[0, 1].set_title('Cell-Level Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Puzzle accuracy
        axes[1, 0].plot(val_epochs, self.metrics['val_puzzle_acc'], 'm-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Puzzle Accuracy')
        axes[1, 0].set_title('Full Puzzle Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Constraint violations
        axes[1, 1].plot(val_epochs, self.metrics['val_violations'], 'orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Average Violations')
        axes[1, 1].set_title('Constraint Violations')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/training_curves.png")
        plt.close(fig)


def main():
    """Run CPU-optimized Sudoku HRM experiment."""
    experiment = CPUSudokuExperiment()
    experiment.run_experiment()


if __name__ == "__main__":
    main()