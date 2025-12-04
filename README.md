# NeuroSpike 🧠⚡

**A comprehensive Spiking Neural Network framework for event-based neuromorphic computing**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

NeuroSpike is a Python framework for building and training Spiking Neural Networks (SNNs) with support for:

- **Leaky Integrate-and-Fire (LIF) Neurons** - Biologically inspired neuron models
- **Event-Based Processing** - Handle DVS/EMU sensor data
- **STDP Learning** - Spike-Timing-Dependent Plasticity
- **Multi-Layer Networks** - Configurable deep architectures
- **Comprehensive Visualization** - Rich plotting utilities

## 🏗️ Architecture

```
NeuroSpike Architecture
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Input Layer          Hidden Layers           Output Layer   │
│  ┌──────────┐        ┌──────────┐            ┌──────────┐  │
│  │  Event   │───────▶│ Spiking  │───────────▶│ Readout  │  │
│  │  Stream  │        │ Neurons  │            │  Layer   │  │
│  │ (DVS/EMU)│        │  (LIF)   │            │ (Spike   │  │
│  └──────────┘        └──────────┘            │  Count)  │  │
│       │                   │                   └──────────┘  │
│       │                   │                        │         │
│       ▼                   ▼                        ▼         │
│  Temporal              Synaptic              Classification  │
│  Encoding              Dynamics              Decision        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib

### Install

```bash
# Clone repository
git clone https://github.com/yourusername/neurospike.git
cd neurospike

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## 🚀 Quick Start

```python
from neurospike import (
    NeuronParams, NetworkConfig,
    NeuroSpikeNetwork, EventStream, Trainer, Visualizer
)

# 1. Configure network
neuron_params = NeuronParams()
config = NetworkConfig(
    input_size=128,
    hidden_sizes=[256, 128],
    output_size=10
)

# 2. Create network
network = NeuroSpikeNetwork(config, neuron_params)

# 3. Generate synthetic event data
event_stream = EventStream(width=16, height=8)
events = event_stream.generate_synthetic_events(pattern='moving_bar')
spike_train = event_stream.encode_to_spike_train(events)

# 4. Train network
trainer = Trainer(network)
train_data = [(spike_train, 0)]  # (input, label) pairs
loss, acc = trainer.train_epoch(train_data, duration=100.0)

# 5. Make predictions
prediction = network.predict(spike_train, duration=100.0)

# 6. Visualize
visualizer = Visualizer()
visualizer.plot_network_activity(network.activity_history)
```

## 📁 Project Structure

```
neurospike/
├── neurospike/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration classes
│   ├── neurons.py           # Neuron models (LIF, Izhikevich)
│   ├── layers.py            # Spiking layers
│   ├── network.py           # Network architecture
│   ├── learning.py          # STDP and training
│   ├── events.py            # Event stream processing
│   └── visualization.py     # Plotting utilities
├── examples/
│   └── demo.py              # Complete demo
├── tests/
│   └── test_all.py          # Unit tests
├── README.md                # This file
├── requirements.txt         # Dependencies
└── setup.py                 # Installation script
```

## 🎯 Features

### Neuron Models

- **LIF (Leaky Integrate-and-Fire)** - Standard spiking neuron
- **Adaptive LIF** - With spike-frequency adaptation
- **Izhikevich** - Rich dynamics neuron model

### Learning Rules

- **STDP** - Spike-Timing-Dependent Plasticity
- **Reward-Modulated STDP** - For reinforcement learning
- **Weight normalization** - Homeostatic plasticity

### Event Processing

- **Synthetic event generation** - Multiple patterns
- **Event encoding** - Convert events to spike trains
- **Temporal/spatial filtering** - Event preprocessing

### Visualization

- Spike raster plots
- Network activity heatmaps
- Weight matrices
- Training curves
- Confusion matrices
- STDP learning windows
- Network architecture diagrams

## 📊 Example Results

### Training Performance

```
Epoch 10/10
  Train: Loss=0.3421, Acc=89.50%
  Test:  Loss=0.4156, Acc=85.00%
```

### Network Statistics

```
Total neurons: 394
Total connections: 65,536
Layer 0: 128 → 256 neurons
  Weight mean: 0.0123
  Active neurons: 243/256
```

## 🔬 Advanced Usage

### Custom Neuron Parameters

```python
custom_params = NeuronParams(
    tau_m=30.0,          # Slower membrane dynamics
    v_thresh=-45.0,      # Lower threshold
    refractory_period=5.0  # Longer refractory
)
```

### Recurrent Networks

```python
from neurospike.network import RecurrentSpikingNetwork

recurrent_net = RecurrentSpikingNetwork(
    config, 
    neuron_params,
    recurrent_prob=0.1  # 10% recurrent connectivity
)
```

### Custom Event Patterns

```python
event_stream = EventStream(width=32, height=32)
events = event_stream.generate_synthetic_events(
    pattern='rotating',
    duration=200.0,
    velocity=1.5
)
```

## 📖 Documentation

### Core Classes

- **`NeuroSpikeNetwork`** - Main network class
- **`LIFNeuron`** - Individual neuron
- **`SpikingLayer`** - Layer of connected neurons
- **`EventStream`** - Event data handler
- **`Trainer`** - Training manager
- **`Visualizer`** - Plotting utilities

### Key Methods

```python
# Forward pass
output = network.forward(spike_train, duration=100.0)

# Make prediction
pred = network.predict(spike_train, duration=100.0)

# Train one epoch
loss, acc = trainer.train_epoch(train_data)

# Evaluate
test_loss, test_acc, metrics = trainer.evaluate(test_data)

# Save/load weights
network.save_weights('model.npy')
network.load_weights('model.npy')
```

## 🧪 Running Examples

### Basic Demo

```bash
cd examples
python demo.py
```

### Run Tests

```bash
python -m pytest tests/
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 References

1. **LIF Neurons**: Gerstner & Kistler (2002) - "Spiking Neuron Models"
2. **STDP**: Bi & Poo (1998) - "Synaptic Modifications in Cultured Hippocampal Neurons"
3. **Event-based Vision**: Gallego et al. (2020) - "Event-based Vision: A Survey"

## 🙏 Acknowledgments

- Inspired by biological neural networks
- Built for neuromorphic computing research
- Designed for event-based sensors (DVS, ATIS, etc.)

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: neurospike@example.com

---

**Built with ❤️ for neuromorphic computing**