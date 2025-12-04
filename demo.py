# examples/demo.py
"""
Complete demonstration of NeuroSpike framework

This script shows how to:
1. Generate synthetic event data
2. Create and configure a spiking neural network
3. Train the network
4. Evaluate performance
5. Visualize results
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append('..')

from neurospike import (
    NeuronParams, NetworkConfig, TrainingConfig,
    NeuroSpikeNetwork, EventStream, Trainer, Visualizer
)


def generate_dataset(num_samples: int = 100, num_classes: int = 10):
    """
    Generate synthetic dataset with different event patterns for each class
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        
    Returns:
        List of (spike_train, label) tuples
    """
    print(f"Generating {num_samples} synthetic samples...")
    
    event_stream = EventStream(width=16, height=8)
    dataset = []
    
    patterns = ['moving_bar', 'random', 'rotating', 'expanding']
    
    for i in range(num_samples):
        # Assign class based on pattern
        class_id = i % num_classes
        pattern = patterns[class_id % len(patterns)]
        
        # Generate events with some variation
        velocity = 0.5 + (class_id * 0.1)
        events = event_stream.generate_synthetic_events(
            pattern=pattern,
            num_events=300,
            duration=100.0,
            velocity=velocity
        )
        
        # Convert to spike train
        spike_train = event_stream.encode_to_spike_train(events, time_window=100.0)
        
        dataset.append((spike_train, class_id))
    
    return dataset


def main():
    """Main demonstration function"""
    
    print("=" * 80)
    print(" " * 25 + "NeuroSpike Framework Demo")
    print("=" * 80)
    
    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================
    print("\n[Step 1] Configuring Network...")
    
    neuron_params = NeuronParams(
        tau_m=20.0,
        tau_s=5.0,
        v_rest=-70.0,
        v_thresh=-50.0,
        v_reset=-75.0,
        refractory_period=2.0
    )
    
    network_config = NetworkConfig(
        input_size=128,
        hidden_sizes=[256, 128],
        output_size=10,
        dt=1.0
    )
    
    training_config = TrainingConfig(
        learning_rate=0.001,
        num_epochs=10,
        duration=50.0
    )
    
    print(f"  ✓ Input size: {network_config.input_size}")
    print(f"  ✓ Hidden layers: {network_config.hidden_sizes}")
    print(f"  ✓ Output size: {network_config.output_size}")
    print(f"  ✓ Neuron type: LIF (tau_m={neuron_params.tau_m}ms)")
    
    # ========================================================================
    # 2. CREATE NETWORK
    # ========================================================================
    print("\n[Step 2] Creating Network...")
    
    network = NeuroSpikeNetwork(network_config, neuron_params)
    print(f"  ✓ Network created successfully")
    print(f"  ✓ Total parameters: {sum(l.input_size * l.output_size for l in network.layers)}")
    print(network)
    
    # ========================================================================
    # 3. GENERATE DATA
    # ========================================================================
    print("\n[Step 3] Generating Training Data...")
    
    train_data = generate_dataset(num_samples=80, num_classes=10)
    test_data = generate_dataset(num_samples=20, num_classes=10)
    
    print(f"  ✓ Training samples: {len(train_data)}")
    print(f"  ✓ Test samples: {len(test_data)}")
    
    # ========================================================================
    # 4. TRAINING
    # ========================================================================
    print("\n[Step 4] Training Network...")
    print("-" * 80)
    
    trainer = Trainer(network, learning_rate=training_config.learning_rate)
    
    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(
            train_data,
            duration=training_config.duration,
            shuffle=True,
            verbose=False
        )
        
        # Evaluate
        test_loss, test_acc, metrics = trainer.evaluate(
            test_data,
            duration=training_config.duration,
            verbose=False
        )
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%}")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2%}")
    
    print("\n" + "-" * 80)
    print("Training Complete!")
    
    # ========================================================================
    # 5. FINAL EVALUATION
    # ========================================================================
    print("\n[Step 5] Final Evaluation...")
    
    test_loss, test_acc, metrics = trainer.evaluate(
        test_data,
        duration=training_config.duration,
        verbose=True
    )
    
    # ========================================================================
    # 6. TEST SINGLE SAMPLE
    # ========================================================================
    print("\n[Step 6] Testing Single Sample...")
    
    test_sample, test_label = test_data[0]
    network.reset()
    prediction = network.predict(test_sample, duration=50.0)
    proba = network.predict_proba(test_sample, duration=50.0)
    
    print(f"  True label: {test_label}")
    print(f"  Predicted: {prediction}")
    print(f"  Confidence: {proba[prediction]:.2%}")
    print(f"  Spike counts: {network.spike_counts}")
    
    # ========================================================================
    # 7. NETWORK STATISTICS
    # ========================================================================
    print("\n[Step 7] Network Statistics...")
    
    stats = network.get_network_statistics()
    print(f"  Total neurons: {stats['total_neurons']}")
    print(f"  Total connections: {stats['total_connections']}")
    
    for layer_stat in stats['layer_stats']:
        print(f"\n  Layer {layer_stat['layer_idx']}:")
        print(f"    Size: {layer_stat['input_size']} → {layer_stat['output_size']}")
        print(f"    Weight mean: {layer_stat['weight_stats']['mean']:.4f}")
        print(f"    Weight std: {layer_stat['weight_stats']['std']:.4f}")
        print(f"    Active neurons: {layer_stat['activity_stats']['active_neurons']}/{layer_stat['output_size']}")
    
    # ========================================================================
    # 8. VISUALIZATIONS
    # ========================================================================
    print("\n[Step 8] Generating Visualizations...")
    
    visualizer = Visualizer()
    
    # 8.1 Training curves
    print("  ✓ Plotting training curves...")
    visualizer.plot_training_curves(
        trainer.train_losses,
        trainer.train_accuracies
    )
    
    # 8.2 Network activity
    print("  ✓ Plotting network activity...")
    layer_names = ['Hidden-1', 'Hidden-2', 'Output']
    visualizer.plot_network_activity(
        network.activity_history,
        layer_names
    )
    
    # 8.3 Weight matrices
    print("  ✓ Plotting weight matrices...")
    for i, layer in enumerate(network.layers):
        visualizer.plot_weight_matrix(
            layer.weights,
            title=f"Layer {i} Weights"
        )
    
    # 8.4 Confusion matrix
    print("  ✓ Plotting confusion matrix...")
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=[f"Class {i}" for i in range(10)]
    )
    
    # 8.5 STDP window
    print("  ✓ Plotting STDP learning window...")
    visualizer.plot_stdp_window()
    
    # 8.6 Network architecture
    print("  ✓ Plotting network architecture...")
    visualizer.plot_network_architecture(network_config.get_layer_sizes())
    
    # ========================================================================
    # 9. SAVE MODEL
    # ========================================================================
    print("\n[Step 9] Saving Model...")
    
    network.save_weights('neurospike_model.npy')
    print("  ✓ Model saved to 'neurospike_model.npy'")
    
    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "=" * 80)
    print(" " * 30 + "Demo Complete!")
    print("=" * 80)
    print("\nShowing all visualizations...")
    plt.show()


if __name__ == "__main__":
    main()