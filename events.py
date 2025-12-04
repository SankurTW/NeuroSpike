# neurospike/events.py
"""
Event stream processing for neuromorphic sensors (DVS, EMU, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class EventPolarity(Enum):
    """Event polarity types"""
    NEGATIVE = 0
    POSITIVE = 1


class EventStream:
    """
    Handles event-based data from neuromorphic sensors
    
    Event-based sensors (Dynamic Vision Sensors) output asynchronous
    events when pixel brightness changes, rather than frames.
    
    Each event contains:
    - (x, y): Pixel coordinates
    - t: Timestamp (microseconds or ms)
    - p: Polarity (ON/OFF or +/-)
    """
    
    def __init__(self, width: int = 128, height: int = 128):
        """
        Initialize event stream
        
        Args:
            width: Sensor width in pixels
            height: Sensor height in pixels
        """
        self.width = width
        self.height = height
        self.events = []
        self.accumulated_frame = np.zeros((height, width))
        
    def add_event(self, x: int, y: int, timestamp: float, polarity: int):
        """
        Add single event to stream
        
        Args:
            x: X coordinate (0 to width-1)
            y: Y coordinate (0 to height-1)
            timestamp: Event timestamp
            polarity: Event polarity (0 or 1)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.events.append({
                'x': x, 'y': y, 't': timestamp, 'p': polarity
            })
    
    def add_events_batch(self, events: np.ndarray):
        """
        Add batch of events
        
        Args:
            events: Array of shape (N, 4) with columns [x, y, t, p]
        """
        for event in events:
            self.add_event(int(event[0]), int(event[1]), event[2], int(event[3]))
    
    def clear(self):
        """Clear all events"""
        self.events = []
        self.accumulated_frame = np.zeros((self.height, self.width))
    
    def get_events_in_window(self, start_time: float, end_time: float) -> List[dict]:
        """
        Get events within time window
        
        Args:
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            List of events in the time window
        """
        return [e for e in self.events if start_time <= e['t'] < end_time]
    
    def encode_to_spike_train(self, events: Optional[np.ndarray] = None, 
                               time_window: float = 100.0,
                               time_bins: Optional[int] = None) -> np.ndarray:
        """
        Convert events to spike train representation
        
        Args:
            events: Event array (N, 4) or None to use stored events
            time_window: Duration of spike train (ms)
            time_bins: Number of time bins (default: int(time_window))
            
        Returns:
            Spike train of shape (num_pixels, time_bins)
        """
        if events is None:
            events = np.array([[e['x'], e['y'], e['t'], e['p']] for e in self.events])
        
        if time_bins is None:
            time_bins = int(time_window)
        
        num_pixels = self.width * self.height
        spike_train = np.zeros((num_pixels, time_bins))
        
        for event in events:
            x, y, t, p = event
            pixel_idx = int(y * self.width + x)
            time_idx = int(t % time_bins)
            
            if pixel_idx < num_pixels and time_idx < time_bins:
                spike_train[pixel_idx, time_idx] += 1
        
        return spike_train
    
    def encode_to_frame(self, events: Optional[np.ndarray] = None,
                        accumulate: bool = True) -> np.ndarray:
        """
        Convert events to 2D frame representation
        
        Args:
            events: Event array or None to use stored events
            accumulate: If True, accumulate events; if False, count only
            
        Returns:
            Frame of shape (height, width)
        """
        if events is None:
            events = np.array([[e['x'], e['y'], e['t'], e['p']] for e in self.events])
        
        frame = np.zeros((self.height, self.width))
        
        for event in events:
            x, y, t, p = event
            x, y = int(x), int(y)
            if 0 <= x < self.width and 0 <= y < self.height:
                if accumulate:
                    frame[y, x] += (1 if p == 1 else -1)
                else:
                    frame[y, x] += 1
        
        return frame
    
    def generate_synthetic_events(self, pattern: str = 'moving_bar', 
                                   duration: float = 100.0, 
                                   num_events: int = 1000,
                                   velocity: float = 1.0) -> np.ndarray:
        """
        Generate synthetic event stream for testing
        
        Args:
            pattern: Pattern type ('moving_bar', 'random', 'rotating', 'expanding')
            duration: Total duration (ms)
            num_events: Number of events to generate
            velocity: Movement velocity for patterns
            
        Returns:
            Event array of shape (num_events, 4)
        """
        events = []
        
        if pattern == 'moving_bar':
            # Vertical bar moving horizontally
            for i in range(num_events):
                t = (i / num_events) * duration
                x = int((t * velocity) % self.width)
                y = np.random.randint(0, self.height)
                p = np.random.choice([0, 1])
                events.append([x, y, t, p])
                
        elif pattern == 'random':
            # Random events uniformly distributed
            for i in range(num_events):
                t = (i / num_events) * duration
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                p = np.random.choice([0, 1])
                events.append([x, y, t, p])
                
        elif pattern == 'rotating':
            # Rotating pattern around center
            cx, cy = self.width // 2, self.height // 2
            for i in range(num_events):
                t = (i / num_events) * duration
                angle = (t * velocity * 2 * np.pi / duration) % (2 * np.pi)
                radius = min(self.width, self.height) // 4
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                x = np.clip(x, 0, self.width - 1)
                y = np.clip(y, 0, self.height - 1)
                p = 1 if np.sin(angle) > 0 else 0
                events.append([x, y, t, p])
                
        elif pattern == 'expanding':
            # Expanding circle from center
            cx, cy = self.width // 2, self.height // 2
            for i in range(num_events):
                t = (i / num_events) * duration
                radius = (t * velocity) % (min(self.width, self.height) // 2)
                angle = np.random.uniform(0, 2 * np.pi)
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                x = np.clip(x, 0, self.width - 1)
                y = np.clip(y, 0, self.height - 1)
                p = 1
                events.append([x, y, t, p])
        
        return np.array(events)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical information about event stream
        
        Returns:
            Dictionary with statistics
        """
        if not self.events:
            return {
                'num_events': 0,
                'duration': 0.0,
                'event_rate': 0.0,
                'polarity_ratio': 0.0
            }
        
        times = [e['t'] for e in self.events]
        polarities = [e['p'] for e in self.events]
        
        return {
            'num_events': len(self.events),
            'duration': max(times) - min(times) if times else 0.0,
            'event_rate': len(self.events) / (max(times) - min(times)) if times else 0.0,
            'polarity_ratio': sum(polarities) / len(polarities) if polarities else 0.0,
            'spatial_coverage': len(set((e['x'], e['y']) for e in self.events)) / (self.width * self.height)
        }


class EventProcessor:
    """Advanced event processing utilities"""
    
    @staticmethod
    def apply_temporal_filter(events: np.ndarray, tau: float = 10.0) -> np.ndarray:
        """
        Apply temporal filtering to events
        
        Args:
            events: Event array
            tau: Time constant for filtering
            
        Returns:
            Filtered events
        """
        if len(events) == 0:
            return events
        
        # Sort by time
        sorted_events = events[events[:, 2].argsort()]
        filtered = []
        
        last_event_time = {}
        
        for event in sorted_events:
            x, y, t, p = event
            key = (int(x), int(y), int(p))
            
            if key not in last_event_time or (t - last_event_time[key]) > tau:
                filtered.append(event)
                last_event_time[key] = t
        
        return np.array(filtered)
    
    @staticmethod
    def apply_spatial_filter(events: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply spatial filtering (remove isolated events)
        
        Args:
            events: Event array
            kernel_size: Size of spatial neighborhood
            
        Returns:
            Filtered events
        """
        if len(events) < 2:
            return events
        
        filtered = []
        for i, event in enumerate(events):
            x, y, t, p = event
            
            # Count neighbors
            neighbors = 0
            for other_event in events:
                ox, oy, ot, op = other_event
                if abs(ox - x) <= kernel_size and abs(oy - y) <= kernel_size:
                    if abs(ot - t) < 10:  # Within 10ms
                        neighbors += 1
            
            if neighbors > 1:  # Keep if has neighbors
                filtered.append(event)
        
        return np.array(filtered)