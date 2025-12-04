# neurospike/layers.py
"""
Spiking neural network layers with synaptic connectivity
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque
from .neurons import LIFNeuron
from .config import NeuronParams


class SpikingLayer:
    """
    Layer of spiking neurons with weighted synaptic connections
    
    Implements:
    - Feed-forward connectivity between layers
    - Synaptic weight matrix
    - Spike history tracking for learning
    - STDP weight updates
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 neuron_params: NeuronParams,
                 weight_init: str = 'xavier'):
        """
        Initialize spiking layer
        
        Args:
            input_size: Number of input connections
            output_size: Number of neurons in layer
            neuron_params: Parameters for LIF neurons
            weight_init: Weight initialization method ('xavier', 'uniform', 'normal')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.neuron_params = neuron_params
        
        # Create neurons
        self.neurons = [LIFNeuron(neuron_params) for _ in range(output_size)]
        
        # Initialize weights
        self.weights = self._initialize_weights(weight_init)
        self.bias = np.zeros(output_size)
        
        # Spike history for STDP (keep last 100 spike times)
        self.pre_spike_times = [deque(maxlen=100) for _ in range(input_size)]
        self.post_spike_times = [deque(maxlen=100) for _ in range(output_size)]
        
        # Activity tracking
        self.spike_counts = np.zeros(output_size)
        self.total_spikes = 0
        
    def _initialize_weights(self, method: str) -> np.ndarray:
        """
        Initialize weight matrix
        
        Args:
            method: Initialization method
            
        Returns:
            Weight matrix of shape (output_size, input_size)
        """
        if method == 'xavier':
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (self.input_size + self.output_size))
            weights = np.random.randn(self.output_size, self.input_size) * scale
        elif method == 'uniform':
            # Uniform initialization [-0.5, 0.5]
            weights = np.random.uniform(-0.5, 0.5, 
                                       (self.output_size, self.input_size))
        elif method == 'normal':
            # Normal initialization with std=0.1
            weights = np.random.randn(self.output_size, self.input_size) * 0.1
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        return weights
    
    def forward(self, input_spikes: np.ndarray, dt: float, 
                time: float) -> np.ndarray:
        """
        Forward propagation through layer
        
        Args:
            input_spikes: Binary spike vector (input_size,)
            dt: Time step size
            time: Current simulation time
            
        Returns:
            Output spike vector (output_size,)
        """
        output_spikes = np.zeros(self.output_size)
        
        # Record pre-synaptic spikes
        for i, spike in enumerate(input_spikes):
            if spike > 0:
                self.pre_spike_times[i].append(time)
        
        # Update each neuron
        for i, neuron in enumerate(self.neurons):
            # Compute weighted input current
            input_current = np.dot(self.weights[i], input_spikes) + self.bias[i]
            
            # Update neuron state
            if neuron.update(input_current, dt, time):
                output_spikes[i] = 1
                self.post_spike_times[i].append(time)
                self.spike_counts[i] += 1
                self.total_spikes [i] += 1
        
        return output_spikes
    
    def apply_stdp(self, learning_rate: float = 0.001,
                   tau_plus: float = 20.0, tau_minus: float = 20.0,
                   a_plus: float = 0.01, a_minus: float = 0.01):
        """
        Apply STDP learning rule to update weights
        
        Args:
            learning_rate: Overall learning rate
            tau_plus: LTP time constant
            tau_minus: LTD time constant
            a_plus: LTP amplitude
            a_minus: LTD amplitude
        """
        for post_idx in range(self.output_size):
            if len(self.post_spike_times[post_idx]) == 0:
                continue
            
            for pre_idx in range(self.input_size):
                if len(self.pre_spike_times[pre_idx]) == 0:
                    continue
                
                # Compute weight changes for all spike pairs
                total_dw = 0.0
                num_pairs = 0
                
                for post_time in self.post_spike_times[post_idx]:
                    for pre_time in self.pre_spike_times[pre_idx]:
                        dt = post_time - pre_time
                        
                        # Only consider spikes within temporal window
                        if abs(dt) < 50.0:  # 50ms window
                            if dt > 0:  # LTP (post after pre)
                                dw = a_plus * np.exp(-dt / tau_plus)
                            else:  # LTD (pre after post)
                                dw = -a_minus * np.exp(dt / tau_minus)
                            
                            total_dw += dw
                            num_pairs += 1
                
                # Apply average weight change
                if num_pairs > 0:
                    avg_dw = total_dw / num_pairs
                    self.weights[post_idx, pre_idx] += learning_rate * avg_dw
        
        # Clip weights to reasonable range
        self.weights = np.clip(self.weights, -1.0, 1.0)
    
    def normalize_weights(self, norm_type: str = 'l2'):
        """
        Normalize weights for each neuron
        
        Args:
            norm_type: Normalization type ('l1', 'l2', 'max')
        """
        for i in range(self.output_size):
            if norm_type == 'l1':
                norm = np.sum(np.abs(self.weights[i]))
            elif norm_type == 'l2':
                norm = np.sqrt(np.sum(self.weights[i]**2))
            elif norm_type == 'max':
                norm = np.max(np.abs(self.weights[i]))
            else:
                raise ValueError(f"Unknown normalization type: {norm_type}")
            
            if norm > 1e-8:
                self.weights[i] /= norm
    
    def get_weight_statistics(self) -> dict:
        """
        Get statistics about layer weights
        
        Returns:
            Dictionary with weight statistics
        """
        return {
            'mean': np.mean(self.weights),
            'std': np.std(self.weights),
            'min': np.min(self.weights),
            'max': np.max(self.weights),
            'sparsity': np.mean(np.abs(self.weights) < 0.01)
        }
    
    def get_activity_statistics(self) -> dict:
        """
        Get statistics about layer activity
        
        Returns:
            Dictionary with activity statistics
        """
        return {
            'total_spikes': self.total_spikes,
            'mean_spike_count': np.mean(self.spike_counts),
            'std_spike_count': np.std(self.spike_counts),
            'max_spike_count': np.max(self.spike_counts),
            'active_neurons': np.sum(self.spike_counts > 0),
            'activity_ratio': np.sum(self.spike_counts > 0) / self.output_size
        }
    
    def reset(self):
        """Reset all neurons and spike history"""
        for neuron in self.neurons:
            neuron.reset()
        
        for i in range(self.input_size):
            self.pre_spike_times[i].clear()
        
        for i in range(self.output_size):
            self.post_spike_times[i].clear()
        
        self.spike_counts = np.zeros(self.output_size)
        self.total_spikes = 0


class ConvolutionalSpikingLayer:
    """
    Convolutional spiking layer for spatial feature extraction
    
    Implements 2D convolutions with spiking neurons for processing
    event-based vision data
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1,
                 neuron_params: NeuronParams = None):
        """
        Initialize convolutional spiking layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output feature maps
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            neuron_params: Parameters for neurons
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        if neuron_params is None:
            neuron_params = NeuronParams()
        
        # Initialize convolutional kernels
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.kernels = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        
        self.bias = np.zeros(out_channels)
        self.neurons = None  # Will be created based on input size
    
    def forward(self, input_spikes: np.ndarray, dt: float, 
                time: float) -> np.ndarray:
        """
        Convolutional forward pass
        
        Args:
            input_spikes: Input spike tensor (in_channels, height, width)
            dt: Time step
            time: Current time
            
        Returns:
            Output spike tensor (out_channels, out_height, out_width)
        """
        in_c, in_h, in_w = input_spikes.shape
        
        # Compute output dimensions
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        
        # Initialize neurons if needed
        if self.neurons is None:
            self.neurons = [
                [LIFNeuron(NeuronParams()) for _ in range(out_w)]
                for _ in range(out_h)
            ]
        
        output_spikes = np.zeros((self.out_channels, out_h, out_w))
        
        # Perform convolution
        for c_out in range(self.out_channels):
            for h_out in range(out_h):
                for w_out in range(out_w):
                    h_start = h_out * self.stride
                    w_start = w_out * self.stride
                    
                    # Extract receptive field
                    receptive_field = input_spikes[
                        :, 
                        h_start:h_start + self.kernel_size,
                        w_start:w_start + self.kernel_size
                    ]
                    
                    # Compute weighted sum
                    current = np.sum(receptive_field * self.kernels[c_out]) + self.bias[c_out]
                    
                    # Update neuron
                    if self.neurons[h_out][w_out].update(current, dt, time):
                        output_spikes[c_out, h_out, w_out] = 1
        
        return output_spikes