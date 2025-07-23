import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from models.hrm import HierarchicalReasoningModel
from training.gradient import DeepSupervisionTrainer
from analysis.visualization import HRMVisualizer


def enhanced_pattern_task():
    """Create a more interesting pattern recognition task"""
    
    def create_advanced_patterns(n_samples, input_dim=200):
        """Create balanced, complex patterns that require reasoning"""
        X = torch.randn(n_samples, 1, input_dim)
        y = []
        
        # Ensure balanced distribution
        samples_per_class = n_samples // 5
        class_counts = [0] * 5
        
        for i in range(n_samples):
            sample = X[i, 0]
            
            # Use simpler, more reliable patterns
            first_half = sample[:input_dim//2]
            second_half = sample[input_dim//2:]
            
            # Assign based on simple statistics, ensuring balance
            if class_counts[0] < samples_per_class and first_half.mean() > 0.2:
                label = 0  # "Positive start"
            elif class_counts[1] < samples_per_class and second_half.mean() > 0.2:
                label = 1  # "Positive end"
            elif class_counts[2] < samples_per_class and sample.std() > 1.2:
                label = 2  # "High variance"
            elif class_counts[3] < samples_per_class and sample.abs().max() > 2.0:
                label = 3  # "Large outlier"
            else:
                # Find least common class for remainder
                label = min(range(5), key=lambda x: class_counts[x])
            
            class_counts[label] += 1
            y.append(label)
        
        return X, torch.tensor(y)
    
    return create_advanced_patterns


def visualized_training_experiment():
    """Run training experiment with comprehensive visualizations"""
    print("=" * 80)
    print("HRM VISUAL TRAINING EXPERIMENT")
    print("=" * 80)
    
    # Setup
    visualizer = HRMVisualizer("visualizations")
    pattern_generator = enhanced_pattern_task()
    
    # Model configuration - simplified for stability
    config = {
        'input_dim': 200,
        'hidden_dim': 128,  # Smaller hidden dimension
        'output_dim': 5,
        'num_transformer_layers': 2,  # Fewer layers
        'num_heads': 4,  # Fewer heads
        'N': 2,  # Fewer cycles
        'T': 3,  # Fewer steps per cycle
    }
    
    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  Total reasoning steps: {config['N'] * config['T']}")
    
    # Create model
    model = HierarchicalReasoningModel(**config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Generate datasets
    print(f"\nGenerating datasets...")
    train_X, train_y = pattern_generator(500)
    val_X, val_y = pattern_generator(100)
    test_X, test_y = pattern_generator(50)
    
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples") 
    print(f"  Test: {len(test_X)} samples")
    
    # Training setup with stability-focused hyperparameters
    trainer = DeepSupervisionTrainer(model, segments_per_example=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for stability
    
    # Training history
    history = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'gradient_norms': {},
        'memory_mb': 0
    }
    
    # Training loop with visualization
    n_epochs = 10  # Fewer epochs to see cleaner patterns
    batch_size = 16  # Smaller batch size
    
    print(f"\nStarting training for {n_epochs} epochs...")
    start_time = time.time()
    
    best_val_acc = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        
        # Shuffle training data
        perm = torch.randperm(len(train_X))
        train_X_shuffled = train_X[perm]
        train_y_shuffled = train_y[perm]
        
        for i in range(0, len(train_X), batch_size):
            batch_X = train_X_shuffled[i:i+batch_size]
            batch_y = train_y_shuffled[i:i+batch_size]
            
            if len(batch_X) == 0:
                continue
                
            metrics = trainer.train_step(batch_X, batch_y, criterion, optimizer, max_grad_norm=0.1)
            epoch_train_loss += metrics['average_loss']
            n_batches += 1
            
            # Store gradient norms (from first batch of epoch)
            if i == 0:  # First batch of epoch
                history['gradient_norms'] = metrics['gradient_info']
                history['memory_mb'] = metrics['memory_info']['param_memory_mb']
        
        avg_train_loss = epoch_train_loss / n_batches if n_batches > 0 else 0
        
        # Validation phase
        model.eval()
        val_metrics = trainer.validate_step(val_X, val_y, criterion)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy'] or 0
        
        # Update history
        history['epochs'].append(epoch + 1)
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{n_epochs}: "
              f"Train Loss: {avg_train_loss:7.4f}, "
              f"Val Loss: {val_loss:7.4f}, "
              f"Val Acc: {val_acc:6.2%}")
        
        # Create visualization every few epochs
        if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:
            print(f"  📊 Generating visualization...")
            
            # Update history with current best metrics
            history['final_train_loss'] = avg_train_loss
            history['final_val_loss'] = val_loss
            history['best_val_acc'] = best_val_acc
            history['total_time'] = time.time() - start_time
            
            # Create dashboard
            dashboard_path = visualizer.create_training_dashboard(
                model, history, test_data=(test_X, test_y)
            )
            print(f"  💾 Saved: {dashboard_path}")
    
    print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    
    # Final test evaluation
    print(f"\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        _, test_output = model(test_X)
        test_predictions = test_output.argmax(dim=-1).squeeze()
        test_accuracy = (test_predictions == test_y).float().mean().item()
    
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Create final prediction visualization
    pred_viz_path = visualizer.create_prediction_visualization(
        model, test_X, test_y, test_predictions, "Advanced_Patterns"
    )
    print(f"Prediction visualization: {pred_viz_path}")
    
    # Pattern analysis
    print(f"\nPattern Distribution:")
    for i in range(5):
        count = (test_y == i).sum().item()
        pred_count = (test_predictions == i).sum().item()
        print(f"  Class {i}: {count} true, {pred_count} predicted")
    
    # Show where visualizations are saved
    viz_dir = Path("visualizations")
    print(f"\n📁 All visualizations saved in: {viz_dir.absolute()}")
    print(f"   Latest dashboard: {viz_dir / 'latest_dashboard.png'}")
    
    return model, history, (test_X, test_y, test_predictions)


def quick_convergence_demo():
    """Quick demo showing convergence behavior"""
    print("\n" + "="*50)
    print("CONVERGENCE BEHAVIOR DEMO")
    print("="*50)
    
    visualizer = HRMVisualizer("visualizations")
    
    # Smaller model for quick demo
    model = HierarchicalReasoningModel(
        input_dim=50,
        hidden_dim=128,
        output_dim=3,
        N=4,
        T=3
    )
    
    # Single sample
    x = torch.randn(1, 1, 50)
    
    print("Analyzing convergence pattern...")
    
    # Track multiple forward passes
    with torch.no_grad():
        states = model.initialize_hidden_states(1, x.device)
        
        for pass_num in range(3):
            print(f"  Forward pass {pass_num + 1}...")
            all_states, output = model(x, states, return_all_steps=True)
            
            # Compute residuals
            residuals = []
            for i in range(1, len(all_states)):
                residual = torch.norm(all_states[i][1] - all_states[i-1][1]).item()
                residuals.append(residual)
            
            print(f"    Final residual: {residuals[-1]:.6f}")
            
            # Use final state for next pass
            states = (all_states[-1][0].detach(), all_states[-1][1].detach())
    
    # Create a mock history for visualization
    mock_history = {
        'epochs': [1],
        'train_losses': [1.0],
        'val_losses': [1.2],
        'gradient_norms': {'input_embedding': 0.1, 'output_head': 0.2},
        'final_train_loss': 1.0,
        'memory_mb': 2.0
    }
    
    viz_path = visualizer.create_training_dashboard(
        model, mock_history, test_data=(x, torch.tensor([0]))
    )
    print(f"Convergence visualization: {viz_path}")


if __name__ == "__main__":
    # Run the full experiment
    try:
        model, history, test_results = visualized_training_experiment()
        
        # Quick convergence demo
        quick_convergence_demo()
        
        print(f"\n🎉 Experiment complete! Check the 'visualizations' folder for:")
        print(f"   - Training dashboards showing model behavior")
        print(f"   - Prediction analysis")
        print(f"   - Convergence patterns")
        print(f"   - State evolution trajectories")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()