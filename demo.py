import torch
import torch.nn as nn
from models.hrm import HierarchicalReasoningModel
from training.gradient import DeepSupervisionTrainer


def simple_demo():
    """Simple demonstration of HRM usage"""
    print("=" * 60)
    print("Hierarchical Reasoning Model Demo")
    print("=" * 60)
    
    # Model configuration
    config = {
        'input_dim': 20,
        'hidden_dim': 128,
        'output_dim': 10,
        'num_transformer_layers': 2,
        'num_heads': 4,
        'N': 2,  # Number of high-level cycles
        'T': 3,  # Low-level steps per cycle
    }
    
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  Total computation steps: {config['N'] * config['T']}")
    
    # Create model
    model = HierarchicalReasoningModel(**config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Create sample data
    batch_size = 4
    seq_len = 1
    x = torch.randn(batch_size, seq_len, config['input_dim'])
    y = torch.randint(0, config['output_dim'], (batch_size,))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Forward pass demo
    print("\n" + "-" * 40)
    print("Forward Pass Demo")
    print("-" * 40)
    
    # Get initial states
    initial_states = model.initialize_hidden_states(batch_size, x.device)
    print(f"Initial state shapes: H={initial_states[0].shape}, L={initial_states[1].shape}")
    
    # Run forward pass
    final_states, output = model(x, initial_states)
    print(f"Output shape: {output.shape}")
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output.squeeze(1), y)
    print(f"Loss: {loss.item():.4f}")
    
    # Training demo
    print("\n" + "-" * 40)
    print("Training Demo with Deep Supervision")
    print("-" * 40)
    
    # Create trainer
    trainer = DeepSupervisionTrainer(model, segments_per_example=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training step
    metrics = trainer.train_step(x, y, criterion, optimizer)
    
    print(f"Training metrics:")
    print(f"  Average loss: {metrics['average_loss']:.4f}")
    print(f"  Segment losses: {[f'{l:.4f}' for l in metrics['segment_losses']]}")
    print(f"  Memory usage: {metrics['memory_info']['param_memory_mb']:.2f} MB")
    
    # Inference demo
    print("\n" + "-" * 40)
    print("Inference Demo")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        # Single sample inference
        x_test = torch.randn(1, 1, config['input_dim'])
        _, output = model(x_test)
        prediction = output.argmax(dim=-1).item()
        print(f"Prediction: {prediction}")
        
        # Get all intermediate states
        all_states, _ = model(x_test, return_all_steps=True)
        print(f"Number of intermediate states: {len(all_states)}")
    
    # Participation ratio demo
    print("\n" + "-" * 40)
    print("Dimensionality Analysis")
    print("-" * 40)
    
    # Collect states from multiple samples
    with torch.no_grad():
        x_batch = torch.randn(20, 1, config['input_dim'])
        states, _ = model(x_batch)
        
        pr_high = model.compute_participation_ratio(states[0])
        pr_low = model.compute_participation_ratio(states[1])
        
        print(f"High-level participation ratio: {pr_high:.2f}")
        print(f"Low-level participation ratio: {pr_low:.2f}")
        print(f"Ratio (H/L): {pr_high/pr_low:.2f}")


def reasoning_task_demo():
    """Demo of a simple reasoning task"""
    print("\n" + "=" * 60)
    print("Simple Pattern Recognition Task")
    print("=" * 60)
    
    # Create a model for pattern recognition
    model = HierarchicalReasoningModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=5,  # 5 pattern classes
        num_transformer_layers=3,
        N=3,
        T=4
    )
    
    # Create synthetic pattern data
    # Pattern: if sum of first half > sum of second half, class 0, else class 1, etc.
    def create_pattern_data(n_samples):
        X = torch.randn(n_samples, 1, 100)
        y = []
        
        for i in range(n_samples):
            sample = X[i, 0]
            first_half_sum = sample[:50].sum()
            second_half_sum = sample[50:].sum()
            
            if first_half_sum > second_half_sum * 1.5:
                label = 0
            elif second_half_sum > first_half_sum * 1.5:
                label = 1
            elif torch.abs(first_half_sum - second_half_sum) < 0.5:
                label = 2
            elif sample.max() > 2.0:
                label = 3
            else:
                label = 4
            
            y.append(label)
        
        return X, torch.tensor(y)
    
    # Generate data
    train_X, train_y = create_pattern_data(100)
    test_X, test_y = create_pattern_data(20)
    
    print("Training on synthetic pattern recognition task...")
    
    # Training
    trainer = DeepSupervisionTrainer(model, segments_per_example=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    n_epochs = 10
    for epoch in range(n_epochs):
        # Mini-batch training
        batch_size = 10
        epoch_loss = 0.0
        
        for i in range(0, len(train_X), batch_size):
            batch_X = train_X[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]
            
            metrics = trainer.train_step(batch_X, batch_y, criterion, optimizer)
            epoch_loss += metrics['average_loss']
        
        # Validation
        val_metrics = trainer.validate_step(test_X, test_y, criterion)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss: {epoch_loss/(len(train_X)/batch_size):.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2%}")
    
    print("\nPattern recognition training complete!")


if __name__ == "__main__":
    simple_demo()
    reasoning_task_demo()