# neurospike/visualization.py
"""
Visualization utilities for spiking neural networks
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Optional, Tuple
import matplotlib.gridspec as gridspec


class Visualizer:
    """
    Comprehensive visualization tools for NeuroSpike networks
    
    Provides methods to visualize:
    - Spike raster plots
    - Network activity over time
    - Weight matrices
    - Training curves
    - Neuron dynamics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('default')
    
    @staticmethod
    def plot_spike_raster(spike_times: Dict[int, List[float]], 
                          title: str = "Spike Raster Plot",
                          figsize: Tuple[int, int] = (12, 6),
                          color: str = 'black',
                          marker_size: int = 50):
        """
        Plot spike raster showing spike times for multiple neurons
        
        Args:
            spike_times: Dict mapping neuron_id to list of spike times
            title: Plot title
            figsize: Figure size
            color: Spike marker color
            marker_size: Size of spike markers
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for neuron_id, times in spike_times.items():
            if len(times) > 0:
                ax.scatter(times, [neuron_id] * len(times), 
                          marker='|', s=marker_size, c=color, alpha=0.8)
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Neuron ID', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_network_activity(activity_history: Dict[int, List[np.ndarray]], 
                             layer_names: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (14, 10),
                             cmap: str = 'hot'):
        """
        Plot activity of all network layers over time
        
        Args:
            activity_history: Dict mapping layer_idx to activity arrays
            layer_names: Names for each layer
            figsize: Figure size
            cmap: Colormap for activity visualization
        """
        num_layers = len(activity_history)
        
        if layer_names is None:
            layer_names = [f'Layer {i+1}' for i in range(num_layers)]
        
        fig, axes = plt.subplots(num_layers, 1, figsize=figsize)
        
        if num_layers == 1:
            axes = [axes]
        
        for layer_idx, ax in enumerate(axes):
            if layer_idx in activity_history and len(activity_history[layer_idx]) > 0:
                activity = np.array(activity_history[layer_idx]).T
                
                im = ax.imshow(activity, aspect='auto', cmap=cmap, 
                              interpolation='nearest', origin='lower')
                
                ax.set_ylabel(f'{layer_names[layer_idx]}\nNeuron ID', fontsize=10)
                ax.set_title(f'{layer_names[layer_idx]} Activity', 
                           fontsize=11, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Spike Count', rotation=270, labelpad=15)
            
            if layer_idx == num_layers - 1:
                ax.set_xlabel('Time Step', fontsize=10)
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_weight_matrix(weights: np.ndarray, 
                          title: str = "Weight Matrix",
                          figsize: Tuple[int, int] = (10, 8),
                          cmap: str = 'RdBu_r',
                          show_colorbar: bool = True):
        """
        Visualize weight matrix
        
        Args:
            weights: Weight matrix (output_size, input_size)
            title: Plot title
            figsize: Figure size
            cmap: Colormap
            show_colorbar: Whether to show colorbar
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Center colormap at zero
        vmax = np.abs(weights).max()
        
        im = ax.imshow(weights, aspect='auto', cmap=cmap,
                      vmin=-vmax, vmax=vmax, interpolation='nearest')
        
        ax.set_xlabel('Input Neuron', fontsize=12)
        ax.set_ylabel('Output Neuron', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Weight Value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_training_curves(losses: List[float], 
                            accuracies: List[float],
                            val_losses: Optional[List[float]] = None,
                            val_accuracies: Optional[List[float]] = None,
                            figsize: Tuple[int, int] = (14, 5)):
        """
        Plot training and validation curves
        
        Args:
            losses: Training losses
            accuracies: Training accuracies
            val_losses: Validation losses (optional)
            val_accuracies: Validation accuracies (optional)
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, losses, 'b-', linewidth=2, label='Train Loss', marker='o')
        if val_losses:
            ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, accuracies, 'g-', linewidth=2, label='Train Acc', marker='o')
        if val_accuracies:
            ax2.plot(epochs, val_accuracies, 'orange', linewidth=2, 
                    label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    @staticmethod
    def plot_neuron_dynamics(voltage_trace: List[float],
                           spike_times: List[float],
                           dt: float = 1.0,
                           figsize: Tuple[int, int] = (12, 5)):
        """
        Plot membrane potential and spikes for a single neuron
        
        Args:
            voltage_trace: Membrane potential over time
            spike_times: Times when neuron spiked
            dt: Time step size
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        time = np.arange(len(voltage_trace)) * dt
        
        # Plot voltage trace
        ax.plot(time, voltage_trace, 'b-', linewidth=1.5, label='Membrane Potential')
        
        # Mark spike times
        if spike_times:
            spike_voltages = [voltage_trace[int(t/dt)] if int(t/dt) < len(voltage_trace) 
                            else voltage_trace[-1] for t in spike_times]
            ax.scatter(spike_times, spike_voltages, c='red', s=100, 
                      marker='^', label='Spikes', zorder=5)
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Membrane Potential (mV)', fontsize=12)
        ax.set_title('Neuron Dynamics', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (10, 8),
                            cmap: str = 'Blues'):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names for each class
            figsize: Figure size
            cmap: Colormap
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(confusion_matrix, cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20)
        
        # Set ticks
        num_classes = confusion_matrix.shape[0]
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, int(confusion_matrix[i, j]),
                             ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max()/2 else "white")
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_stdp_window(tau_plus: float = 20.0, tau_minus: float = 20.0,
                        a_plus: float = 0.01, a_minus: float = 0.01,
                        figsize: Tuple[int, int] = (10, 6)):
        """
        Plot STDP learning window
        
        Args:
            tau_plus: LTP time constant
            tau_minus: LTD time constant
            a_plus: LTP amplitude
            a_minus: LTD amplitude
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Time differences
        dt = np.linspace(-50, 50, 200)
        
        # STDP function
        dw = np.where(dt > 0,
                     a_plus * np.exp(-dt / tau_plus),
                     -a_minus * np.exp(dt / tau_minus))
        
        ax.plot(dt, dw, 'b-', linewidth=2.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Shade regions
        ax.fill_between(dt[dt > 0], 0, dw[dt > 0], alpha=0.3, color='green', label='LTP')
        ax.fill_between(dt[dt < 0], 0, dw[dt < 0], alpha=0.3, color='red', label='LTD')
        
        ax.set_xlabel('Δt = t_post - t_pre (ms)', fontsize=12)
        ax.set_ylabel('Weight Change (Δw)', fontsize=12)
        ax.set_title('STDP Learning Window', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_network_architecture(layer_sizes: List[int],
                                 figsize: Tuple[int, int] = (12, 6)):
        """
        Visualize network architecture
        
        Args:
            layer_sizes: List of layer sizes
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        num_layers = len(layer_sizes)
        max_neurons = max(layer_sizes)
        
        # Spacing
        h_spacing = 1.0 / (num_layers + 1)
        
        for layer_idx, size in enumerate(layer_sizes):
            x = (layer_idx + 1) * h_spacing
            v_spacing = 0.8 / (size + 1)
            
            # Draw neurons
            for neuron_idx in range(min(size, 10)):  # Limit display
                y = 0.1 + (neuron_idx + 1) * v_spacing
                circle = plt.Circle((x, y), 0.02, color='steelblue', zorder=2)
                ax.add_patch(circle)
                
                # Draw connections to next layer
                if layer_idx < num_layers - 1:
                    next_size = min(layer_sizes[layer_idx + 1], 10)
                    next_x = (layer_idx + 2) * h_spacing
                    next_v_spacing = 0.8 / (layer_sizes[layer_idx + 1] + 1)
                    
                    for next_neuron_idx in range(next_size):
                        next_y = 0.1 + (next_neuron_idx + 1) * next_v_spacing
                        ax.plot([x, next_x], [y, next_y], 'k-', 
                               alpha=0.1, linewidth=0.5, zorder=1)
            
            # Layer label
            ax.text(x, 0.95, f'Layer {layer_idx}\n({size} neurons)',
                   ha='center', va='top', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Network Architecture', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig, ax