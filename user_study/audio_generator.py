"""
Audio signal generation and playback utilities for UUPL user study.
Uses the generateSignal6params function to create audio signals.
"""

import numpy as np
import sys
import os
import tempfile
from scipy.io import wavfile
import threading
import time

# Add parent directory to access audiosignal module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audiosignal.generateSignal6params import generate_tone_signal

# Try to import audio playback libraries
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Audio playback disabled.")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class AudioGenerator:
    """Generate and play audio signals for user preference learning."""
    
    def __init__(self):
        """Initialize audio generator with fixed parameters."""
        # Fixed parameters as specified in requirements
        self.duration = 4
        self.cycles = 1
        self.fs = 44100
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'amplitude': [30, 60],
            'frequency': [25, 75],
            'density': [10, 90],
            'gradient': [-50, 50]
        }
        
        # Initialize pygame for audio playback if available
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=self.fs, size=-16, channels=1, buffer=512)
                self.audio_backend = 'pygame'
            except:
                self.audio_backend = None
        elif SOUNDDEVICE_AVAILABLE:
            self.audio_backend = 'sounddevice'
        else:
            self.audio_backend = None
            print("Warning: No audio backend available")
    
    def generate_signal(self, amplitude, frequency, density, gradient):
        """
        Generate audio signal with given parameters.
        
        Parameters:
        -----------
        amplitude : float
            Amplitude parameter [30-60]
        frequency : float
            Frequency parameter [25-75]
        density : float
            Density parameter [10-90]
        gradient : float
            Gradient parameter [-50-50]
            
        Returns:
        --------
        time : ndarray
            Time vector
        data : ndarray
            Audio signal data
        metadata : dict
            Additional metadata for plotting/analysis
        """
        # Validate parameters
        amplitude = np.clip(amplitude, *self.param_ranges['amplitude'])
        frequency = np.clip(frequency, *self.param_ranges['frequency'])
        density = np.clip(density, *self.param_ranges['density'])
        gradient = np.clip(gradient, *self.param_ranges['gradient'])
        
        # Generate signal using the provided function
        time, data, for_plot = generate_tone_signal(
            filler_amplitude=amplitude,
            filler_frequency=frequency,
            filler_density=density,
            filler_env_gradient=gradient,
            duration=self.duration,
            cycles=self.cycles,
            fs=self.fs
        )
        
        # Normalize audio data to prevent clipping
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data)) * 0.8
        
        metadata = {
            'parameters': {
                'amplitude': amplitude,
                'frequency': frequency,
                'density': density,
                'gradient': gradient
            },
            'duration': self.duration,
            'fs': self.fs,
            'for_plot': for_plot
        }
        
        return time, data, metadata
    
    def save_audio(self, data, filename=None):
        """
        Save audio data to WAV file.
        
        Parameters:
        -----------
        data : ndarray
            Audio signal data
        filename : str, optional
            Output filename. If None, creates temporary file.
            
        Returns:
        --------
        filename : str
            Path to saved audio file
        """
        if filename is None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            filename = temp_file.name
            temp_file.close()
        
        # Convert to 16-bit integer
        audio_int16 = (data * 32767).astype(np.int16)
        
        # Save to WAV file
        wavfile.write(filename, self.fs, audio_int16)
        
        return filename
    
    def play_audio(self, data, blocking=True):
        """
        Play audio signal.
        
        Parameters:
        -----------
        data : ndarray
            Audio signal data
        blocking : bool
            Whether to block until playback finishes
            
        Returns:
        --------
        success : bool
            Whether playback was successful
        """
        if self.audio_backend is None:
            print("No audio backend available for playback")
            return False
        
        try:
            if self.audio_backend == 'pygame':
                return self._play_pygame(data, blocking)
            elif self.audio_backend == 'sounddevice':
                return self._play_sounddevice(data, blocking)
        except Exception as e:
            print(f"Audio playback error: {e}")
            return False
        
        return False
    
    def _play_pygame(self, data, blocking):
        """Play audio using pygame."""
        try:
            # Save to temporary file and play
            temp_file = self.save_audio(data)
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            if blocking:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
                
            return True
        except Exception as e:
            print(f"Pygame playback error: {e}")
            return False
    
    def _play_sounddevice(self, data, blocking):
        """Play audio using sounddevice."""
        try:
            if blocking:
                sd.play(data, self.fs)
                sd.wait()
            else:
                sd.play(data, self.fs)
            return True
        except Exception as e:
            print(f"Sounddevice playback error: {e}")
            return False
    
    def stop_audio(self):
        """Stop any currently playing audio."""
        try:
            if self.audio_backend == 'pygame':
                pygame.mixer.music.stop()
            elif self.audio_backend == 'sounddevice':
                sd.stop()
        except:
            pass
    
    def calculate_audio_features(self, data):
        """
        Calculate audio features for preference evaluation.
        
        Parameters:
        -----------
        data : ndarray
            Audio signal data
            
        Returns:
        --------
        features : dict
            Dictionary of calculated features
        """
        features = {}
        
        # Basic statistics
        features['rms'] = np.sqrt(np.mean(data**2))
        features['peak'] = np.max(np.abs(data))
        features['energy'] = np.sum(data**2)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(data)
        
        # Spectral features using FFT
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/self.fs)
        magnitude = np.abs(fft)
        
        # Find peak frequency
        peak_idx = np.argmax(magnitude[:len(magnitude)//2])
        features['peak_frequency'] = freqs[peak_idx]
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        
        return features


def test_audio_generator():
    """Test the audio generator functionality."""
    print("Testing Audio Generator...")
    
    generator = AudioGenerator()
    
    # Test signal generation with different parameters
    test_params = [
        [45, 50, 50, 0],    # Middle values
        [30, 25, 10, -50],  # Minimum values
        [60, 75, 90, 50],   # Maximum values
    ]
    
    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}: Parameters {params}")
        
        try:
            time_vec, signal, metadata = generator.generate_signal(*params)
            print(f"Generated signal: duration={len(signal)/generator.fs:.2f}s, "
                  f"max_amplitude={np.max(np.abs(signal)):.3f}")
            
            # Calculate features
            features = generator.calculate_audio_features(signal)
            print(f"Features: RMS={features['rms']:.3f}, "
                  f"Peak_freq={features['peak_frequency']:.1f}Hz")
            
            # Save audio file
            filename = f"/tmp/test_audio_{i+1}.wav"
            generator.save_audio(signal, filename)
            print(f"Saved to: {filename}")
            
            # Test playback (non-blocking)
            print("Testing playback...")
            success = generator.play_audio(signal, blocking=False)
            print(f"Playback {'successful' if success else 'failed'}")
            
            time.sleep(1)  # Wait a bit between tests
            
        except Exception as e:
            print(f"Error in test {i+1}: {e}")
    
    print("\nAudio generator test completed!")


if __name__ == "__main__":
    test_audio_generator()