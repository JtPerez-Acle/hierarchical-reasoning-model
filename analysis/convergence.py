import torch
import matplotlib.pyplot as plt
import numpy as np
from models.hrm import HierarchicalReasoningModel


def track_convergence(model, x, num_repeats=5):
    """Track how states converge over multiple forward passes"""
    batch_size = x.shape[0]
    device = x.device
    
    # Initialize states
    states = model.initialize_hidden_states(batch_size, device)
    
    # Track residuals
    low_residuals = []
    high_residuals = []
    
    # Run multiple times from same initial state
    prev_low = states[1].clone()
    prev_high = states[0].clone()
    
    for i in range(num_repeats):
        # Get all intermediate states
        all_states, _ = model(x, states, return_all_steps=True)
        
        # Track residuals for each step
        step_low_residuals = []
        step_high_residuals = []
        
        for j, (h_state, l_state) in enumerate(all_states):
            # Compute residuals (change from previous state)
            if j > 0:
                l_residual = torch.norm(l_state - prev_low, dim=-1).mean().item()
                h_residual = torch.norm(h_state - prev_high, dim=-1).mean().item()
                
                step_low_residuals.append(l_residual)
                step_high_residuals.append(h_residual)
                
                prev_low = l_state.clone()
                prev_high = h_state.clone()
        
        low_residuals.append(step_low_residuals)
        high_residuals.append(step_high_residuals)
        
        # Use final state as initial for next iteration
        states = (all_states[-1][0].detach(), all_states[-1][1].detach())
    
    return low_residuals, high_residuals


def plot_convergence_patterns():
    """Visualize the hierarchical convergence behavior"""
    # Create model
    model = HierarchicalReasoningModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=10,
        num_transformer_layers=3,
        N=3,  # 3 high-level cycles
        T=4   # 4 low-level steps per cycle
    )
    
    # Create input
    x = torch.randn(1, 1, 100)
    
    # Track convergence
    low_residuals, high_residuals = track_convergence(model, x, num_repeats=3)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot low-level residuals
    for i, residuals in enumerate(low_residuals):
        steps = range(1, len(residuals) + 1)
        ax1.plot(steps, residuals, 'o-', label=f'Forward pass {i+1}', alpha=0.7)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Low-level Residual (L2 norm)')
    ax1.set_title('Low-level Module Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot high-level residuals
    for i, residuals in enumerate(high_residuals):
        steps = range(1, len(residuals) + 1)
        ax2.plot(steps, residuals, 's-', label=f'Forward pass {i+1}', alpha=0.7)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('High-level Residual (L2 norm)')
    ax2.set_title('High-level Module Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add vertical lines for high-level updates
    for ax in [ax1, ax2]:
        for i in range(1, model.N + 1):
            ax.axvline(x=i * model.T, color='red', linestyle='--', alpha=0.3, 
                      label='H-update' if i == 1 else '')
    
    plt.tight_layout()
    plt.savefig('convergence_patterns.png', dpi=150)
    plt.show()


def analyze_state_evolution():
    """Analyze how states evolve during processing"""
    model = HierarchicalReasoningModel(
        input_dim=50,
        hidden_dim=128,
        output_dim=10,
        N=2,
        T=3
    )
    
    x = torch.randn(5, 1, 50)  # Batch of 5
    
    # Get all states
    all_states, _ = model(x, return_all_steps=True)
    
    # Compute state statistics
    print("State Evolution Analysis")
    print("=" * 50)
    print(f"Total steps: {len(all_states)}")
    print(f"High-level updates at steps: {[i for i in range(model.T, len(all_states)+1, model.T)]}")
    
    # Track norm evolution
    low_norms = []
    high_norms = []
    
    for i, (h_state, l_state) in enumerate(all_states):
        l_norm = torch.norm(l_state, dim=-1).mean().item()
        h_norm = torch.norm(h_state, dim=-1).mean().item()
        
        low_norms.append(l_norm)
        high_norms.append(h_norm)
        
        print(f"\nStep {i+1}:")
        print(f"  L-norm: {l_norm:.4f}")
        print(f"  H-norm: {h_norm:.4f}")
        
        if i > 0:
            # Check if high-level was updated
            h_changed = not torch.allclose(all_states[i][0], all_states[i-1][0])
            print(f"  H-updated: {'Yes' if h_changed else 'No'}")
    
    # Plot norm evolution
    plt.figure(figsize=(8, 6))
    steps = range(1, len(all_states) + 1)
    plt.plot(steps, low_norms, 'o-', label='Low-level norm', color='blue')
    plt.plot(steps, high_norms, 's-', label='High-level norm', color='red')
    
    # Mark high-level updates
    for i in range(model.T, len(all_states) + 1, model.T):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.3)
    
    plt.xlabel('Step')
    plt.ylabel('State Norm')
    plt.title('State Norm Evolution During Processing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('state_evolution.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Analyzing HRM Convergence Patterns...")
    plot_convergence_patterns()
    print("\nAnalyzing State Evolution...")
    analyze_state_evolution()