# neurospike/network.py
"""
Complete spiking neural network architecture
"""

import numpy as np
from typing import List, Dict, Optional
from .config import NetworkConfig, NeuronParams
from .layers import SpikingLayer


class NeuroSpikeNetwork:
    """
    Multi-layer spiking neural network with temporal processing
    
    Architecture:
        Input → Hidden Layer(s) → Output Layer → Readout
        
    The network processes spike trains through multiple layers,
    with each layer performing temporal integration and generating
    output spikes based on LIF neuron dynamics.
    """
    
    def __init__(self, config: NetworkConfig, neuron_params: NeuronParams):
        """
        Initialize NeuroSpike network
        
        Args:
            config: Network architecture configuration
            neuron_params: Parameters for all neurons
        """
        self.config = config
        self.neuron_params = neuron_params
        
        # Build network layers
        self.layers = []
        layer_sizes = config.get_layer_sizes()
        
        for i in range(len(layer_sizes) - 1):
            layer = SpikingLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                neuron_params=neuron_params
            )
            self.layers.append(layer)
        
        # Output readout (spike counting)
        self.spike_counts = np.zeros(config.output_size)
        self.output_history = []
        
        # Activity tracking for visualization
        self.activity_history = {i: [] for i in range(len(self.layers))}
        self.input_history = []
        
    def forward(self, input_spikes: np.ndarray, duration: float,
                record_activity: bool = True) -> np.ndarray:
        """
        Forward propagation through entire network
        
        Args:
            input_spikes: Input spike train (input_size, time_bins) or (input_size,)
            duration: Simulation duration (ms)
            record_activity: Whether to record layer activities
            
        Returns:
            Spike counts for each output neuron
        """
        num_steps = int(duration / self.config.dt)
        self.spike_counts = np.zeros(self.config.output_size)
        
        # Clear activity history if recording
        if record_activity:
            self.activity_history = {i: [] for i in range(len(self.layers))}
            self.input_history = []
            self.output_history = []
        
        # Simulation loop
        for step in range(num_steps):
            time = step * self.config.dt
            
            # Get input for current time step
            if input_spikes.ndim == 2:
                # Temporal input (input_size, time_bins)
                time_idx = min(step, input_spikes.shape[1] - 1)
                current_input = input_spikes[:, time_idx]
            else:
                # Static input (input_size,)
                current_input = input_spikes
            
            if record_activity:
                self.input_history.append(current_input.copy())
            
            # Propagate through layers
            layer_input = current_input
            for layer_idx, layer in enumerate(self.layers):
                layer_output = layer.forward(layer_input, self.config.dt, time)
                
                if record_activity:
                    self.activity_history[layer_idx].append(layer_output.copy())
                
                layer_input = layer_output
            
            # Accumulate output spikes
            self.spike_counts += layer_input
            if record_activity:
                self.output_history.append(layer_input.copy())
        
        return self.spike_counts
    
    def predict(self, input_spikes: np.ndarray, duration: float) -> int:
        """
        Make classification prediction
        
        Args:
            input_spikes: Input spike train
            duration: Simulation duration
            
        Returns:
            Predicted class index
        """
        output = self.forward(input_spikes, duration, record_activity=False)
        return int(np.argmax(output))
    
    def predict_proba(self, input_spikes: np.ndarray, 
                      duration: float) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            input_spikes: Input spike train
            duration: Simulation duration
            
        Returns:
            Probability distribution over classes
        """
        output = self.forward(input_spikes, duration, record_activity=False)
        return self.softmax(output)
    
    @staticmethod
    def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Softmax activation function
        
        Args:
            x: Input array
            temperature: Temperature parameter for softmax
            
        Returns:
            Probability distribution
        """
        x = x / temperature
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def get_layer_activity(self, layer_idx: int) -> np.ndarray:
        """
        Get activity matrix for specific layer
        
        Args:
            layer_idx: Index of layer
            
        Returns:
            Activity matrix (neurons, time_steps)
        """
        if layer_idx not in self.activity_history:
            return np.array([])
        
        return np.array(self.activity_history[layer_idx]).T
    
    def get_network_statistics(self) -> Dict:
        """
        Get statistics for entire network
        
        Returns:
            Dictionary with network statistics
        """
        stats = {
            'num_layers': len(self.layers),
            'total_neurons': sum(layer.output_size for layer in self.layers),
            'total_connections': sum(
                layer.input_size * layer.output_size for layer in self.layers
            ),
            'layer_stats': []
        }
        
        for i, layer in enumerate(self.layers):
            layer_stats = {
                'layer_idx': i,
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'weight_stats': layer.get_weight_statistics(),
                'activity_stats': layer.get_activity_statistics()
            }
            stats['layer_stats'].append(layer_stats)
        
        return stats
    
    def save_weights(self, filepath: str):
        """
        Save network weights to file
        
        Args:
            filepath: Path to save weights
        """
        weights_data = {
            'config': {
                'input_size': self.config.input_size,
                'hidden_sizes': self.config.hidden_sizes,
                'output_size': self.config.output_size,
                'dt': self.config.dt
            },
            'neuron_params': {
                'tau_m': self.neuron_params.tau_m,
                'tau_s': self.neuron_params.tau_s,
                'v_rest': self.neuron_params.v_rest,
                'v_reset': self.neuron_params.v_reset,
                'v_thresh': self.neuron_params.v_thresh,
                'refractory_period': self.neuron_params.refractory_period
            },
            'layers': []
        }
        
        for layer in self.layers:
            weights_data['layers'].append({
                'weights': layer.weights,
                'bias': layer.bias
            })
        
        np.save(filepath, weights_data)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """
        Load network weights from file
        
        Args:
            filepath: Path to load weights from
        """
        weights_data = np.load(filepath, allow_pickle=True).item()
        
        # Verify architecture matches
        config = weights_data['config']
        if (config['input_size'] != self.config.input_size or
            config['output_size'] != self.config.output_size):
            raise ValueError("Network architecture mismatch!")
        
        # Load weights for each layer
        for i, layer_data in enumerate(weights_data['layers']):
            self.layers[i].weights = layer_data['weights']
            self.layers[i].bias = layer_data['bias']
        
        print(f"Weights loaded from {filepath}")
    
    def reset(self):
        """Reset all layers and clear history"""
        for layer in self.layers:
            layer.reset()
        
        self.spike_counts = np.zeros(self.config.output_size)
        self.activity_history = {i: [] for i in range(len(self.layers))}
        self.input_history = []
        self.output_history = []
    
    def __repr__(self) -> str:
        """String representation of network"""
        layer_info = []
        layer_sizes = self.config.get_layer_sizes()
        
        for i in range(len(self.layers)):
            layer_info.append(
                f"Layer {i}: {layer_sizes[i]} → {layer_sizes[i+1]} neurons"
            )
        
        return (
            f"NeuroSpikeNetwork(\n"
            f"  Architecture:\n    " + "\n    ".join(layer_info) + "\n"
            f"  Total parameters: {sum(layer.input_size * layer.output_size for layer in self.layers)}\n"
            f"  Time step: {self.config.dt} ms\n"
            f")"
        )


class RecurrentSpikingNetwork(NeuroSpikeNetwork):
    """
    Recurrent spiking neural network with feedback connections
    
    Extends NeuroSpikeNetwork with recurrent connections within layers
    for temporal memory and dynamics.
    """
    
    def __init__(self, config: NetworkConfig, neuron_params: NeuronParams,
                 recurrent_prob: float = 0.1):
        """
        Initialize recurrent network
        
        Args:
            config: Network configuration
            neuron_params: Neuron parameters
            recurrent_prob: Probability of recurrent connections
        """
        super().__init__(config, neuron_params)
        self.recurrent_prob = recurrent_prob
        
        # Add recurrent weight matrices for each layer
        self.recurrent_weights = []
        for layer in self.layers:
            # Sparse recurrent connectivity
            rec_weights = np.random.randn(layer.output_size, layer.output_size) * 0.1
            mask = np.random.rand(layer.output_size, layer.output_size) > recurrent_prob
            rec_weights[mask] = 0
            np.fill_diagonal(rec_weights, 0)  # No self-connections
            self.recurrent_weights.append(rec_weights)
    
    def forward(self, input_spikes: np.ndarray, duration: float,
                record_activity: bool = True) -> np.ndarray:
        """
        Forward pass with recurrent connections
        
        Includes feedback from previous time step
        """
        num_steps = int(duration / self.config.dt)
        self.spike_counts = np.zeros(self.config.output_size)
        
        # Store previous layer outputs for recurrence
        prev_outputs = [np.zeros(layer.output_size) for layer in self.layers]
        
        for step in range(num_steps):
            time = step * self.config.dt
            
            # Get input
            if input_spikes.ndim == 2:
                time_idx = min(step, input_spikes.shape[1] - 1)
                current_input = input_spikes[:, time_idx]
            else:
                current_input = input_spikes
            
            # Propagate through layers with recurrence
            layer_input = current_input
            for layer_idx, layer in enumerate(self.layers):
                # Add recurrent feedback
                recurrent_input = np.dot(self.recurrent_weights[layer_idx], 
                                        prev_outputs[layer_idx])
                combined_input = layer_input + recurrent_input * 0.5
                
                # Forward through layer
                layer_output = layer.forward(combined_input, self.config.dt, time)
                
                prev_outputs[layer_idx] = layer_output
                layer_input = layer_output
            
            self.spike_counts += layer_input
        
        return self.spike_counts