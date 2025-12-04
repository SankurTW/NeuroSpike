# neurospike/config.py
"""
Configuration classes for NeuroSpike framework
"""

from dataclasses import dataclass
from typing import List

@dataclass
class NeuronParams:
    """
    Parameters for Leaky Integrate-and-Fire (LIF) neurons
    
    Attributes:
        tau_m: Membrane time constant (ms)
        tau_s: Synaptic time constant (ms)
        v_rest: Resting membrane potential (mV)
        v_reset: Reset potential after spike (mV)
        v_thresh: Spike threshold potential (mV)
        refractory_period: Refractory period duration (ms)
    """
    tau_m: float = 20.0
    tau_s: float = 5.0
    v_rest: float = -70.0
    v_reset: float = -75.0
    v_thresh: float = -50.0
    refractory_period: float = 2.0
    
    def __post_init__(self):
        """Validate parameters"""
        assert self.tau_m > 0, "Membrane time constant must be positive"
        assert self.tau_s > 0, "Synaptic time constant must be positive"
        assert self.v_reset < self.v_thresh, "Reset potential must be less than threshold"
        assert self.refractory_period >= 0, "Refractory period cannot be negative"


@dataclass
class NetworkConfig:
    """
    Configuration for NeuroSpike network architecture
    
    Attributes:
        input_size: Number of input neurons
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output neurons
        dt: Simulation time step (ms)
    """
    input_size: int = 128
    hidden_sizes: List[int] = None
    output_size: int = 10
    dt: float = 1.0
    
    def __post_init__(self):
        """Set default values and validate"""
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128]
            
        assert self.input_size > 0, "Input size must be positive"
        assert self.output_size > 0, "Output size must be positive"
        assert self.dt > 0, "Time step must be positive"
        assert all(size > 0 for size in self.hidden_sizes), "All layer sizes must be positive"
    
    def get_layer_sizes(self) -> List[int]:
        """Return complete list of layer sizes including input and output"""
        return [self.input_size] + self.hidden_sizes + [self.output_size]
    
    def num_layers(self) -> int:
        """Return total number of layers (excluding input)"""
        return len(self.hidden_sizes) + 1


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters
    
    Attributes:
        learning_rate: Learning rate for STDP
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        duration: Simulation duration per sample (ms)
    """
    learning_rate: float = 0.001
    num_epochs: int = 10
    batch_size: int = 32
    duration: float = 100.0
    
    def __post_init__(self):
        """Validate training parameters"""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.duration > 0, "Duration must be positive"