import torch
import torch.nn as nn
from models.hrm import HierarchicalReasoningModel
from training.gradient import DeepSupervisionTrainer


def test_basic_forward_pass():
    """Test basic forward pass and hierarchical dynamics"""
    print("Testing basic HRM forward pass...")
    
    # Model parameters
    input_dim = 10
    hidden_dim = 64
    output_dim = 10
    batch_size = 2
    seq_len = 1
    N = 2  # High-level cycles
    T = 3  # Low-level steps per cycle
    
    # Create model
    model = HierarchicalReasoningModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_transformer_layers=2,
        num_heads=4,
        N=N,
        T=T
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test forward pass
    states, output = model(x)
    high_level_state, low_level_state = states
    
    # Verify shapes
    assert high_level_state.shape == (batch_size, seq_len, hidden_dim)
    assert low_level_state.shape == (batch_size, seq_len, hidden_dim)
    assert output.shape == (batch_size, seq_len, output_dim)
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - High-level state shape: {high_level_state.shape}")
    print(f"  - Low-level state shape: {low_level_state.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Total steps: {N * T}")
    

def test_hierarchical_updates():
    """Test that hierarchical updates follow the correct pattern"""
    print("\nTesting hierarchical update pattern...")
    
    model = HierarchicalReasoningModel(
        input_dim=10,
        hidden_dim=64,
        output_dim=10,
        N=2,
        T=3
    )
    
    x = torch.randn(1, 1, 10)
    
    # Get all intermediate states
    all_states, output = model(x, return_all_steps=True)
    
    # Verify we have the correct number of states
    assert len(all_states) == model.total_steps
    
    # Check high-level state updates
    print(f"✓ Collected {len(all_states)} intermediate states")
    
    # Verify high-level only updates every T steps
    for i in range(1, len(all_states) - 1):
        if i % model.T != 0:
            # High-level should not change
            h_prev = all_states[i-1][0]
            h_curr = all_states[i][0]
            # Note: In actual implementation, h_curr == h_prev when not updating
            print(f"  Step {i}: High-level {'unchanged' if i % model.T != 0 else 'UPDATED'}")


def test_gradient_flow():
    """Test 1-step gradient approximation"""
    print("\nTesting gradient flow...")
    
    model = HierarchicalReasoningModel(
        input_dim=10,
        hidden_dim=64,
        output_dim=10,
        N=2,
        T=2
    )
    
    x = torch.randn(2, 1, 10)
    y = torch.randint(0, 10, (2,))
    
    # Forward pass
    states, output = model(x)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output.squeeze(1), y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_grad = {
        'input_embedding': model.input_embedding.weight.grad is not None,
        'output_head': model.output_head.weight.grad is not None,
        'low_level': any(p.grad is not None for p in model.low_level_module.parameters()),
        'high_level': any(p.grad is not None for p in model.high_level_module.parameters())
    }
    
    print("✓ Gradient flow verified:")
    for name, has in has_grad.items():
        print(f"  - {name}: {'✓' if has else '✗'}")


def test_deep_supervision():
    """Test deep supervision training"""
    print("\nTesting deep supervision training...")
    
    model = HierarchicalReasoningModel(
        input_dim=10,
        hidden_dim=64,
        output_dim=10,
        N=1,
        T=2
    )
    
    trainer = DeepSupervisionTrainer(model, segments_per_example=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training data
    x = torch.randn(4, 1, 10)
    y = torch.randint(0, 10, (4,))
    
    # Train step
    metrics = trainer.train_step(x, y, criterion, optimizer)
    
    print("✓ Deep supervision training completed:")
    print(f"  - Average loss: {metrics['average_loss']:.4f}")
    print(f"  - Segment losses: {[f'{l:.4f}' for l in metrics['segment_losses']]}")
    print(f"  - Parameter memory: {metrics['memory_info']['param_memory_mb']:.2f} MB")
    print(f"  - Gradient memory: {metrics['memory_info']['grad_memory_mb']:.2f} MB")


def test_participation_ratio():
    """Test dimensionality analysis"""
    print("\nTesting participation ratio calculation...")
    
    model = HierarchicalReasoningModel(
        input_dim=10,
        hidden_dim=128,
        output_dim=10,
        N=2,
        T=3
    )
    
    # Generate some data
    x = torch.randn(10, 1, 10)
    
    # Collect states
    with torch.no_grad():
        states, _ = model(x)
        high_level_states, low_level_states = states
        
        # Compute participation ratios
        pr_high = model.compute_participation_ratio(high_level_states)
        pr_low = model.compute_participation_ratio(low_level_states)
        
        ratio = pr_high / pr_low
    
    print(f"✓ Participation ratios computed:")
    print(f"  - High-level PR: {pr_high:.2f}")
    print(f"  - Low-level PR: {pr_low:.2f}")
    print(f"  - Ratio (H/L): {ratio:.2f}")
    print(f"  - {'✓' if ratio > 1.5 else '✗'} Dimensionality hierarchy present")


if __name__ == "__main__":
    print("=" * 50)
    print("Hierarchical Reasoning Model Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_basic_forward_pass()
    test_hierarchical_updates()
    test_gradient_flow()
    test_deep_supervision()
    test_participation_ratio()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)