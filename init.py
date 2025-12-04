
"""
NeuroSpike - Spiking Neural Network Framework
A comprehensive framework for event-based neuromorphic computing
"""

__version__ = "1.0.0"
__author__ = "NeuroSpike Team"

from .config import NeuronParams, NetworkConfig
from .neurons import LIFNeuron
from .layers import SpikingLayer
from .network import NeuroSpikeNetwork
from .learning import STDP, Trainer
from .events import EventStream
from .visualization import Visualizer

__all__ = [
    'NeuronParams',
    'NetworkConfig',
    'LIFNeuron',
    'SpikingLayer',
    'NeuroSpikeNetwork',
    'STDP',
    'Trainer',
    'EventStream',
    'Visualizer',
]