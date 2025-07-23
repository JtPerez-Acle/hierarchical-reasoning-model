import torch
import torch.nn as nn
from typing import Tuple, Optional


class OneStepGradientTrainer:
    """Implements 1-step gradient approximation for HRM training
    
    Key insight: Instead of backpropagating through all N*T steps,
    we only compute gradients through the final step, achieving O(1) memory.
    
    The gradient path is:
    output_head -> final_H_state -> final_L_state -> input_embedding
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def verify_gradient_flow(self, loss: torch.Tensor) -> dict:
        """Verify that gradient flow follows the expected path
        
        Returns dict with gradient magnitudes for each component
        """
        grad_info = {}
        
        # Check gradients for key components
        if hasattr(self.model, 'output_head') and self.model.output_head.weight.grad is not None:
            grad_info['output_head'] = self.model.output_head.weight.grad.norm().item()
        else:
            grad_info['output_head'] = 0.0
        
        # Check high-level module gradients
        h_grad_norm = 0.0
        h_params_count = 0
        for layer in self.model.high_level_module:
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    h_grad_norm += param.grad.norm().item()
                    h_params_count += 1
        grad_info['high_level'] = h_grad_norm / max(h_params_count, 1)
        
        # Check low-level module gradients
        l_grad_norm = 0.0
        l_params_count = 0
        for layer in self.model.low_level_module:
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    l_grad_norm += param.grad.norm().item()
                    l_params_count += 1
        grad_info['low_level'] = l_grad_norm / max(l_params_count, 1)
        
        # Check input embedding gradients
        if hasattr(self.model, 'input_embedding') and self.model.input_embedding.weight.grad is not None:
            grad_info['input_embedding'] = self.model.input_embedding.weight.grad.norm().item()
        else:
            grad_info['input_embedding'] = 0.0
        
        return grad_info
    
    def compute_memory_usage(self) -> dict:
        """Estimate memory usage to verify O(1) complexity
        
        Returns dict with memory estimates
        """
        memory_info = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        memory_info['total_parameters'] = total_params
        
        # Estimate gradient memory (should be same as parameters)
        grad_params = sum(p.numel() for p in self.model.parameters() if p.grad is not None)
        memory_info['gradient_parameters'] = grad_params
        
        # Memory in MB (assuming float32)
        memory_info['param_memory_mb'] = (total_params * 4) / (1024 * 1024)
        memory_info['grad_memory_mb'] = (grad_params * 4) / (1024 * 1024)
        
        return memory_info
    
    @staticmethod
    def detach_states(states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detach states to prevent gradient flow through previous segments"""
        return tuple(s.detach() for s in states)


class DeepSupervisionTrainer:
    """Implements deep supervision training for HRM
    
    Key features:
    - Segments training into multiple forward passes
    - Detaches states between segments
    - Accumulates gradients across segments
    """
    
    def __init__(self, model: nn.Module, segments_per_example: int = 4):
        self.model = model
        self.segments_per_example = segments_per_example
        self.gradient_trainer = OneStepGradientTrainer(model)
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        initial_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_grad_norm: float = 1.0
    ) -> dict:
        """Single training step with deep supervision
        
        Returns:
            Dictionary with training metrics
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        if initial_states is None:
            states = self.model.initialize_hidden_states(batch_size, device)
        else:
            states = initial_states
        
        total_loss = 0.0
        segment_losses = []
        
        # Clear gradients at start
        optimizer.zero_grad()
        
        # Train through segments
        for segment in range(self.segments_per_example):
            # Forward pass (with 1-step gradient)
            states, output = self.model(x, states)
            
            # Compute loss (scale down to prevent accumulation explosion)
            loss = criterion(output.squeeze(1), y) / self.segments_per_example
            
            # Backward pass
            loss.backward()
            
            # Detach states for next segment (critical!)
            states = self.gradient_trainer.detach_states(states)
            
            # Track metrics
            total_loss += loss.item() * self.segments_per_example  # Unscale for logging
            segment_losses.append(loss.item() * self.segments_per_example)
        
        # Clip gradients to prevent explosion
        if max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_grad_norm
            )
        else:
            grad_norm = 0.0
        
        # Get gradient info before parameter update
        gradient_info = self.gradient_trainer.verify_gradient_flow(loss)
        
        # Update parameters after all segments
        optimizer.step()
        
        # Compile metrics
        metrics = {
            'total_loss': total_loss,
            'average_loss': total_loss / self.segments_per_example,
            'segment_losses': segment_losses,
            'gradient_info': gradient_info,
            'memory_info': self.gradient_trainer.compute_memory_usage(),
            'grad_norm': grad_norm if max_grad_norm > 0 else 0.0
        }
        
        return metrics
    
    def validate_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module
    ) -> dict:
        """Validation step without gradient computation"""
        with torch.no_grad():
            # Run full model
            _, output = self.model(x)
            loss = criterion(output.squeeze(1), y)
            
            # Compute accuracy if classification
            if len(y.shape) == 1:  # Classification task
                predictions = output.argmax(dim=-1).squeeze(1)
                accuracy = (predictions == y).float().mean().item()
            else:
                accuracy = None
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }