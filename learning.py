# neurospike/learning.py
"""
Learning rules and training algorithms for spiking neural networks
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .network import NeuroSpikeNetwork


class STDP:
    """
    Spike-Timing-Dependent Plasticity (STDP) learning rule
    
    STDP adjusts synaptic weights based on the relative timing of 
    pre- and post-synaptic spikes:
    - If pre-spike comes before post-spike (causal): strengthen synapse (LTP)
    - If post-spike comes before pre-spike (anti-causal): weaken synapse (LTD)
    
    Weight change:
        Δw = A+ * exp(-Δt/τ+)  if Δt > 0 (LTP)
        Δw = -A- * exp(Δt/τ-)  if Δt < 0 (LTD)
    """
    
    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0,
                 a_plus: float = 0.01, a_minus: float = 0.01,
                 w_min: float = -1.0, w_max: float = 1.0):
        """
        Initialize STDP learning rule
        
        Args:
            tau_plus: Time constant for LTP (ms)
            tau_minus: Time constant for LTD (ms)
            a_plus: Maximum LTP amplitude
            a_minus: Maximum LTD amplitude
            w_min: Minimum weight value
            w_max: Maximum weight value
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
    
    def compute_weight_change(self, dt: float) -> float:
        """
        Compute weight change for given spike time difference
        
        Args:
            dt: Time difference (post - pre) in ms
            
        Returns:
            Weight change value
        """
        if dt > 0:  # Post after pre (LTP - strengthen)
            return self.a_plus * np.exp(-dt / self.tau_plus)
        elif dt < 0:  # Pre after post (LTD - weaken)
            return -self.a_minus * np.exp(dt / self.tau_minus)
        else:
            return 0.0
    
    def apply_bounds(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply weight bounds
        
        Args:
            weights: Weight matrix
            
        Returns:
            Bounded weights
        """
        return np.clip(weights, self.w_min, self.w_max)
    
    def get_learning_window(self, dt_range: Tuple[float, float] = (-50, 50),
                           num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get STDP learning window curve
        
        Args:
            dt_range: Range of time differences to plot
            num_points: Number of points to sample
            
        Returns:
            Tuple of (time_diffs, weight_changes)
        """
        dts = np.linspace(dt_range[0], dt_range[1], num_points)
        dws = np.array([self.compute_weight_change(dt) for dt in dts])
        return dts, dws


class RewardModulatedSTDP(STDP):
    """
    Reward-modulated STDP for reinforcement learning
    
    Weight updates are gated by a reward signal:
        Δw = R * STDP(Δt)
    where R is the reward signal
    """
    
    def __init__(self, *args, reward_decay: float = 0.95, **kwargs):
        """
        Initialize R-STDP
        
        Args:
            reward_decay: Decay factor for eligibility trace
        """
        super().__init__(*args, **kwargs)
        self.reward_decay = reward_decay
        self.eligibility_trace = None
    
    def compute_modulated_change(self, dt: float, reward: float) -> float:
        """
        Compute reward-modulated weight change
        
        Args:
            dt: Spike time difference
            reward: Reward signal
            
        Returns:
            Modulated weight change
        """
        base_change = self.compute_weight_change(dt)
        return reward * base_change


class Trainer:
    """
    Training utilities for NeuroSpike networks
    
    Handles training loops, evaluation, and logging
    """
    
    def __init__(self, network: NeuroSpikeNetwork, 
                 learning_rate: float = 0.001,
                 stdp_params: Optional[Dict] = None):
        """
        Initialize trainer
        
        Args:
            network: NeuroSpike network to train
            learning_rate: Learning rate for weight updates
            stdp_params: Parameters for STDP learning rule
        """
        self.network = network
        self.learning_rate = learning_rate
        
        # Initialize STDP
        if stdp_params is None:
            stdp_params = {}
        self.stdp = STDP(**stdp_params)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_times = []
        
    def train_step(self, input_spikes: np.ndarray, target: int, 
                   duration: float) -> Tuple[float, np.ndarray]:
        """
        Single training step
        
        Args:
            input_spikes: Input spike train
            target: Target class label
            duration: Simulation duration
            
        Returns:
            Tuple of (loss, output)
        """
        # Forward pass
        output = self.network.forward(input_spikes, duration)
        
        # Apply STDP to all layers
        for layer in self.network.layers:
            layer.apply_stdp(self.learning_rate)
        
        # Compute loss (negative log likelihood)
        probs = self.network.softmax(output)
        loss = -np.log(probs[target] + 1e-10)
        
        return loss, output
    
    def train_epoch(self, train_data: List[Tuple[np.ndarray, int]], 
                    duration: float = 100.0,
                    shuffle: bool = True,
                    verbose: bool = True) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_data: List of (input, target) tuples
            duration: Simulation duration per sample
            shuffle: Whether to shuffle data
            verbose: Print progress
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        if shuffle:
            indices = np.random.permutation(len(train_data))
            train_data = [train_data[i] for i in indices]
        
        epoch_losses = []
        correct = 0
        total = 0
        
        for i, (input_spikes, target) in enumerate(train_data):
            # Reset network state
            self.network.reset()
            
            # Training step
            loss, output = self.train_step(input_spikes, target, duration)
            
            # Track metrics
            epoch_losses.append(loss)
            pred = np.argmax(output)
            if pred == target:
                correct += 1
            total += 1
            
            # Print progress
            if verbose and (i + 1) % max(1, len(train_data) // 10) == 0:
                current_acc = correct / total
                print(f"  [{i+1}/{len(train_data)}] "
                      f"Loss: {np.mean(epoch_losses):.4f}, "
                      f"Acc: {current_acc:.2%}")
        
        avg_loss = np.mean(epoch_losses)
        accuracy = correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, int]], 
                 duration: float = 100.0,
                 verbose: bool = True) -> Tuple[float, float, Dict]:
        """
        Evaluate on test data
        
        Args:
            test_data: List of (input, target) tuples
            duration: Simulation duration per sample
            verbose: Print results
            
        Returns:
            Tuple of (loss, accuracy, metrics_dict)
        """
        test_losses = []
        correct = 0
        total = 0
        predictions = []
        targets = []
        
        for input_spikes, target in test_data:
            self.network.reset()
            
            # Forward pass only (no learning)
            output = self.network.forward(input_spikes, duration)
            
            # Compute loss
            probs = self.network.softmax(output)
            loss = -np.log(probs[target] + 1e-10)
            test_losses.append(loss)
            
            # Track predictions
            pred = np.argmax(output)
            predictions.append(pred)
            targets.append(target)
            
            if pred == target:
                correct += 1
            total += 1
        
        avg_loss = np.mean(test_losses)
        accuracy = correct / total
        
        # Compute confusion matrix
        num_classes = self.network.config.output_size
        confusion_matrix = np.zeros((num_classes, num_classes))
        for pred, target in zip(predictions, targets):
            confusion_matrix[target, pred] += 1
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'predictions': predictions,
            'targets': targets
        }
        
        if verbose:
            print(f"\nTest Results:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Correct: {correct}/{total}")
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_data: List[Tuple[np.ndarray, int]], 
              val_data: Optional[List[Tuple[np.ndarray, int]]] = None,
              num_epochs: int = 10,
              duration: float = 100.0,
              early_stopping: bool = False,
              patience: int = 3) -> Dict:
        """
        Complete training loop
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            num_epochs: Number of epochs to train
            duration: Simulation duration per sample
            early_stopping: Use early stopping
            patience: Epochs to wait before stopping
            
        Returns:
            Dictionary with training history
        """
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(train_data)}")
        if val_data:
            print(f"Validation samples: {len(val_data)}")
        print("-" * 70)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_data, duration, shuffle=True, verbose=True
            )
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            
            # Validate
            if val_data:
                val_loss, val_acc, _ = self.evaluate(
                    val_data, duration, verbose=False
                )
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
                
                # Early stopping
                if early_stopping:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                            break
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }