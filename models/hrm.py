import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .components import TransformerBlock, truncated_lecun_normal_


class HierarchicalReasoningModel(nn.Module):
    """Hierarchical Reasoning Model (HRM)
    
    Brain-inspired architecture with two modules operating at different timescales:
    - Low-level module (L): Fast, detailed computations (updates every step)
    - High-level module (H): Slow, abstract planning (updates every T steps)
    
    Key features:
    - Hierarchical convergence prevents premature convergence
    - 1-step gradient approximation for O(1) memory training
    - Deep supervision with detached states between segments
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 10,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        N: int = 2,  # Number of high-level cycles
        T: int = 2,  # Low-level steps per cycle
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.N = N
        self.T = T
        self.total_steps = N * T
        
        # Input embedding network (f_I)
        self.input_embedding = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Low-level module (f_L) - Fast transformer encoder
        self.low_level_module = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim * 4,
                max_seq_len=max_seq_len,
                dropout=dropout
            ) for _ in range(num_transformer_layers)
        ])
        
        # High-level module (f_H) - Slow transformer encoder
        self.high_level_module = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim * 4,
                max_seq_len=max_seq_len,
                dropout=dropout
            ) for _ in range(num_transformer_layers)
        ])
        
        # Projection layers for module interaction
        # Low-level receives input from high-level
        self.low_level_h_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # High-level receives input from low-level
        self.high_level_l_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output network (f_O)
        self.output_head = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using truncated LeCun normal distribution"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                truncated_lecun_normal_(module.weight)
    
    def initialize_hidden_states(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states for both modules
        
        Returns:
            Tuple of (high_level_state, low_level_state)
        """
        # Initialize with truncated normal distribution
        high_level_state = torch.empty(
            batch_size, 1, self.hidden_dim, device=device
        )
        low_level_state = torch.empty(
            batch_size, 1, self.hidden_dim, device=device
        )
        
        # Truncated normal initialization (std=1, truncation=2)
        nn.init.trunc_normal_(high_level_state, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(low_level_state, std=1.0, a=-2.0, b=2.0)
        
        return high_level_state, low_level_state
    
    def low_level_step(
        self,
        low_level_state: torch.Tensor,
        high_level_state: torch.Tensor,
        input_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Single step of low-level module
        
        z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, x_tilde; theta_L)
        """
        # Combine inputs: current state + high-level influence + input
        high_influence = self.low_level_h_proj(high_level_state)
        combined = low_level_state + high_influence + input_embedding
        
        # Pass through transformer layers
        for layer in self.low_level_module:
            combined = layer(combined)
        
        return combined
    
    def high_level_step(
        self,
        high_level_state: torch.Tensor,
        low_level_state: torch.Tensor
    ) -> torch.Tensor:
        """Single step of high-level module
        
        z_H^i = f_H(z_H^{i-1}, z_L^{i-1}; theta_H)
        """
        # Combine inputs: current state + low-level information
        low_influence = self.high_level_l_proj(low_level_state)
        combined = high_level_state + low_influence
        
        # Pass through transformer layers
        for layer in self.high_level_module:
            combined = layer(combined)
        
        return combined
    
    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_all_steps: bool = False
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass with hierarchical dynamics
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            initial_states: Optional tuple of (high_level_state, low_level_state)
            return_all_steps: If True, return states from all steps
            
        Returns:
            - Final states: (high_level_state, low_level_state)
            - Output predictions [batch_size, seq_len, output_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states if not provided
        if initial_states is None:
            high_level_state, low_level_state = self.initialize_hidden_states(
                batch_size, device
            )
        else:
            high_level_state, low_level_state = initial_states
        
        # Embed input
        input_embedding = self.input_embedding(x)
        
        # Store all steps if requested
        if return_all_steps:
            all_states = []
        
        # Main hierarchical dynamics loop
        # All but the last step are computed without gradients
        with torch.no_grad():
            for step in range(self.total_steps - 1):
                # Low-level always updates
                low_level_state = self.low_level_step(
                    low_level_state, high_level_state, input_embedding
                )
                
                # High-level updates every T steps
                if (step + 1) % self.T == 0:
                    high_level_state = self.high_level_step(
                        high_level_state, low_level_state
                    )
                
                if return_all_steps:
                    all_states.append((high_level_state.clone(), low_level_state.clone()))
        
        # Final step with gradients (1-step gradient approximation)
        low_level_state = self.low_level_step(
            low_level_state, high_level_state, input_embedding
        )
        high_level_state = self.high_level_step(
            high_level_state, low_level_state
        )
        
        # Generate output from high-level state
        output = self.output_head(high_level_state)
        
        final_states = (high_level_state, low_level_state)
        
        if return_all_steps:
            all_states.append(final_states)
            return all_states, output
        
        return final_states, output
    
    def compute_participation_ratio(
        self, 
        states: torch.Tensor,
        regularization: float = 1e-6
    ) -> float:
        """Compute participation ratio for dimensionality analysis
        
        PR = (Σλ_i)² / Σλ_i²
        
        Where λ_i are eigenvalues of the covariance matrix
        """
        try:
            # Flatten batch and sequence dimensions
            states_flat = states.view(-1, self.hidden_dim)
            
            if states_flat.shape[0] < 2:
                return float(self.hidden_dim)  # Return max possible PR
            
            # Compute covariance matrix with regularization
            centered = states_flat - states_flat.mean(dim=0, keepdim=True)
            cov = torch.matmul(centered.T, centered) / (states_flat.shape[0] - 1)
            
            # Add regularization to diagonal for numerical stability
            cov += regularization * torch.eye(self.hidden_dim, device=cov.device)
            
            # Compute eigenvalues with error handling
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > regularization]  # Remove tiny values
            
            if len(eigenvalues) == 0:
                return 1.0  # Fallback
            
            # Compute participation ratio
            sum_eig = eigenvalues.sum()
            sum_eig_sq = (eigenvalues ** 2).sum()
            
            if sum_eig_sq == 0:
                return 1.0
            
            pr = (sum_eig ** 2) / sum_eig_sq
            
            return pr.item()
            
        except Exception as e:
            # Fallback: use variance-based approximation
            var = states.var(dim=[0, 1])  # Variance across batch and sequence
            effective_dim = (var.sum() ** 2) / (var ** 2).sum()
            return effective_dim.item()


class HRMWithACT(HierarchicalReasoningModel):
    """HRM with Adaptive Computation Time
    
    Extends base HRM with Q-learning based halting mechanism
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Q-head for halt/continue decisions
        self.q_head = nn.Linear(self.hidden_dim, 2, bias=False)  # [Q_halt, Q_continue]
        
        # Epsilon for exploration
        self.epsilon = 0.1
    
    def forward_with_act(
        self,
        x: torch.Tensor,
        initial_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_segments: int = 10,
        training: bool = True
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass with adaptive computation time
        
        Returns:
            - Final states
            - Output predictions
            - Halting decisions (for ACT loss)
        """
        batch_size = x.shape[0]
        device = x.device
        
        states = initial_states
        halting_decisions = []
        
        for segment in range(max_segments):
            # Run one segment
            states, output = self.forward(x, states)
            
            # Compute Q-values for halting decision
            q_values = self.q_head(states[0])  # Use high-level state
            
            if training:
                # Epsilon-greedy exploration
                if torch.rand(1).item() < self.epsilon:
                    halt = torch.rand(batch_size, 1, 1, device=device) > 0.5
                else:
                    halt = q_values.argmax(dim=-1, keepdim=True) == 0
            else:
                # Greedy selection during inference
                halt = q_values.argmax(dim=-1, keepdim=True) == 0
            
            halting_decisions.append((q_values, halt))
            
            # Check if all sequences have halted
            if halt.all():
                break
            
            # Detach states for next segment
            states = tuple(s.detach() for s in states)
        
        return states, output, halting_decisions