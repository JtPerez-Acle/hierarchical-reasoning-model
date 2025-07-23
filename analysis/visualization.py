import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import os
from pathlib import Path
from models.hrm import HierarchicalReasoningModel
from typing import List, Tuple, Dict, Optional


class HRMVisualizer:
    """Comprehensive visualization system for HRM training and analysis"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.run_counter = self._get_next_run_number()
        
    def _get_next_run_number(self) -> int:
        """Get the next run number for unique filenames"""
        existing_runs = list(self.save_dir.glob("run_*.png"))
        if not existing_runs:
            return 1
        numbers = [int(f.stem.split('_')[1]) for f in existing_runs if f.stem.split('_')[1].isdigit()]
        return max(numbers) + 1 if numbers else 1
    
    def create_training_dashboard(
        self,
        model: HierarchicalReasoningModel,
        training_history: Dict,
        test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> str:
        """Create a comprehensive dashboard showing training progress and model behavior"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title with run information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f'HRM Training Dashboard - Run #{self.run_counter} - {timestamp}', fontsize=16)
        
        # 1. Training Loss Curve
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_curves(ax1, training_history)
        
        # 2. Gradient Flow Visualization
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_gradient_flow(ax2, training_history)
        
        # 3. State Evolution
        ax3 = fig.add_subplot(gs[1, :2])
        if test_data:
            self._plot_state_evolution(ax3, model, test_data[0])
        
        # 4. Convergence Pattern
        ax4 = fig.add_subplot(gs[1, 2:])
        if test_data:
            self._plot_convergence_pattern(ax4, model, test_data[0])
        
        # 5. Dimensionality Analysis
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_dimensionality_analysis(ax5, model, test_data[0] if test_data else None)
        
        # 6. Attention Patterns (placeholder)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_attention_heatmap(ax6, model, test_data[0] if test_data else None)
        
        # 7. Performance Metrics
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_performance_metrics(ax7, training_history)
        
        # 8. Model Architecture Info
        ax8 = fig.add_subplot(gs[2, 3])
        self._plot_model_info(ax8, model)
        
        # Save with unique filename
        filename = f"run_{self.run_counter:04d}_dashboard.png"
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save a "latest" version for easy access
        latest_path = self.save_dir / "latest_dashboard.png"
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        
        self.run_counter += 1
        return str(filepath)
    
    def _plot_training_curves(self, ax, history):
        """Plot training and validation loss curves"""
        epochs = history.get('epochs', [])
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        
        if epochs and train_losses:
            ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            if val_losses:
                ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No training data yet', ha='center', va='center')
            ax.set_title('Training Progress')
    
    def _plot_gradient_flow(self, ax, history):
        """Visualize gradient flow through model components"""
        gradient_norms = history.get('gradient_norms', {})
        
        if gradient_norms and any(v > 0 for v in gradient_norms.values()):
            components = list(gradient_norms.keys())
            values = [max(v, 1e-10) for v in gradient_norms.values()]  # Prevent log(0)
            
            # Create bars with different colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(components)]
            bars = ax.bar(components, values, color=colors)
            
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Flow Analysis')
            
            # Only use log scale if we have meaningful positive values
            if max(values) > 1e-8:
                try:
                    ax.set_yscale('log')
                except:
                    pass  # Fall back to linear scale
            
            # Add value labels on bars
            for bar, val in zip(bars, gradient_norms.values()):
                if val > 1e-10:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No gradient data available\n(or all gradients are zero)', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Gradient Flow Analysis')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    def _plot_state_evolution(self, ax, model, x):
        """Show how states evolve during forward pass"""
        with torch.no_grad():
            all_states, _ = model(x[:1], return_all_steps=True)
        
        steps = len(all_states)
        low_norms = []
        high_norms = []
        
        for h_state, l_state in all_states:
            low_norms.append(torch.norm(l_state, dim=-1).mean().item())
            high_norms.append(torch.norm(h_state, dim=-1).mean().item())
        
        x_axis = range(1, steps + 1)
        ax.plot(x_axis, low_norms, 'b-o', label='Low-level', linewidth=2)
        ax.plot(x_axis, high_norms, 'r-s', label='High-level', linewidth=2)
        
        # Mark high-level updates
        for i in range(model.T, steps + 1, model.T):
            ax.axvline(x=i, color='red', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Computation Step')
        ax.set_ylabel('State Norm')
        ax.set_title('State Evolution During Forward Pass')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_pattern(self, ax, model, x):
        """Visualize convergence behavior"""
        with torch.no_grad():
            states = model.initialize_hidden_states(1, x.device)
            
            residuals = []
            for _ in range(3):  # Multiple forward passes
                all_states, _ = model(x[:1], states, return_all_steps=True)
                
                step_residuals = []
                prev_state = all_states[0]
                
                for curr_state in all_states[1:]:
                    l_residual = torch.norm(curr_state[1] - prev_state[1]).item()
                    step_residuals.append(l_residual)
                    prev_state = curr_state
                
                residuals.append(step_residuals)
                states = (all_states[-1][0].detach(), all_states[-1][1].detach())
        
        # Plot residuals
        for i, res in enumerate(residuals):
            ax.plot(range(1, len(res) + 1), res, 'o-', label=f'Pass {i+1}', alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Residual (L2 norm)')
        ax.set_title('Convergence Pattern')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_dimensionality_analysis(self, ax, model, x):
        """Show participation ratio analysis"""
        if x is None:
            x = torch.randn(20, 1, model.input_embedding.in_features)
        
        with torch.no_grad():
            states, _ = model(x)
            pr_high = model.compute_participation_ratio(states[0])
            pr_low = model.compute_participation_ratio(states[1])
        
        # Create bar plot
        modules = ['Low-level', 'High-level']
        prs = [pr_low, pr_high]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax.bar(modules, prs, color=colors)
        
        # Add value labels
        for bar, pr in zip(bars, prs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{pr:.1f}', ha='center', va='bottom')
        
        # Add ratio annotation
        ratio = pr_high / pr_low
        ax.text(0.5, max(prs) * 0.8, f'Ratio (H/L): {ratio:.2f}',
               ha='center', transform=ax.transData)
        
        ax.set_ylabel('Participation Ratio')
        ax.set_title('Dimensionality Analysis')
        ax.set_ylim(0, max(prs) * 1.3)
    
    def _plot_attention_heatmap(self, ax, model, x):
        """Placeholder for attention visualization"""
        # This would show attention patterns if we extracted them
        ax.text(0.5, 0.5, 'Attention Patterns\n(Future Feature)', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Attention Analysis')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _plot_performance_metrics(self, ax, history):
        """Show key performance metrics"""
        metrics_text = []
        
        if 'final_train_loss' in history:
            metrics_text.append(f"Final Train Loss: {history['final_train_loss']:.4f}")
        if 'final_val_loss' in history:
            metrics_text.append(f"Final Val Loss: {history['final_val_loss']:.4f}")
        if 'best_val_acc' in history:
            metrics_text.append(f"Best Val Accuracy: {history['best_val_acc']:.2%}")
        if 'total_time' in history:
            metrics_text.append(f"Training Time: {history['total_time']:.1f}s")
        if 'memory_mb' in history:
            metrics_text.append(f"Memory Usage: {history['memory_mb']:.1f} MB")
        
        if not metrics_text:
            metrics_text = ["No metrics available yet"]
        
        # Display metrics
        y_pos = 0.9
        for text in metrics_text:
            ax.text(0.1, y_pos, text, fontsize=12, transform=ax.transAxes)
            y_pos -= 0.15
        
        ax.set_title('Performance Metrics')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _plot_model_info(self, ax, model):
        """Display model configuration"""
        info_text = [
            f"Hidden Dim: {model.hidden_dim}",
            f"N (H-cycles): {model.N}",
            f"T (L-steps): {model.T}",
            f"Total Steps: {model.total_steps}",
            f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
        ]
        
        y_pos = 0.9
        for text in info_text:
            ax.text(0.1, y_pos, text, fontsize=11, transform=ax.transAxes)
            y_pos -= 0.15
        
        ax.set_title('Model Configuration')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def create_prediction_visualization(
        self,
        model: HierarchicalReasoningModel,
        x: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        task_name: str = "Task"
    ) -> str:
        """Visualize model predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Prediction accuracy
        ax = axes[0, 0]
        correct = (y_pred == y_true).float()
        accuracy = correct.mean().item()
        
        ax.bar(['Correct', 'Incorrect'], 
               [correct.sum().item(), len(correct) - correct.sum().item()],
               color=['green', 'red'])
        ax.set_title(f'{task_name} Predictions (Acc: {accuracy:.2%})')
        ax.set_ylabel('Count')
        
        # 2. Confusion matrix (if applicable)
        ax = axes[0, 1]
        if len(torch.unique(y_true)) <= 10:  # Only for small number of classes
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
                im = ax.imshow(cm, cmap='Blues')
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                plt.colorbar(im, ax=ax)
            except ImportError:
                ax.text(0.5, 0.5, 'Confusion Matrix\n(sklearn not available)', 
                       ha='center', va='center')
                ax.set_title('Confusion Matrix')
        else:
            ax.text(0.5, 0.5, 'Too many classes\nfor confusion matrix', 
                   ha='center', va='center')
            ax.set_title('Confusion Matrix')
        
        # 3. State trajectory (PCA)
        ax = axes[1, 0]
        with torch.no_grad():
            all_states, _ = model(x[:10], return_all_steps=True)
            
            # Collect all high-level states
            h_states = torch.cat([s[0] for s in all_states], dim=0)
            h_states_flat = h_states.view(-1, model.hidden_dim)
            
            # Simple 2D projection
            U, S, V = torch.svd(h_states_flat.T)
            projected = torch.matmul(h_states_flat, U[:, :2])
            
            # Plot trajectories
            n_samples = min(5, x.shape[0])
            for i in range(n_samples):
                traj = projected[i::x.shape[0]]
                ax.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), 'o-', alpha=0.6)
        
        ax.set_title('State Trajectories (PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        
        # 4. Processing depth analysis
        ax = axes[1, 1]
        with torch.no_grad():
            # Run with different N values
            depths = [1, 2, 3, 4]
            accuracies = []
            
            for n in depths:
                if n <= model.N:
                    # Temporarily modify model depth
                    original_N = model.N
                    model.N = n
                    model.total_steps = n * model.T
                    
                    _, output = model(x)
                    pred = output.argmax(dim=-1).squeeze()
                    acc = (pred == y_true).float().mean().item()
                    accuracies.append(acc)
                    
                    model.N = original_N
                    model.total_steps = original_N * model.T
        
        if accuracies:
            ax.plot(depths[:len(accuracies)], accuracies, 'o-', linewidth=2)
            ax.set_xlabel('N (High-level cycles)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Performance vs Depth')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = f"predictions_{task_name.lower()}_{self.run_counter:04d}.png"
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        return str(filepath)


def create_animated_training_gif(image_dir: str, output_path: str = "training_animation.gif"):
    """Create an animated GIF from training visualizations"""
    try:
        from PIL import Image
        import glob
        
        # Get all dashboard images
        images = []
        for filename in sorted(glob.glob(os.path.join(image_dir, "run_*.png"))):
            images.append(Image.open(filename))
        
        if images:
            # Save as animated GIF
            images[0].save(output_path, save_all=True, append_images=images[1:], 
                          duration=500, loop=0)
            print(f"Created animation: {output_path}")
    except ImportError:
        print("PIL not installed, skipping GIF creation")