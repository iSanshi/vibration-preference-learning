# 4D Audio Preference Learning System (UUPL)

A complete implementation of Uncertainty-aware User Preference Learning (UUPL) for audio signal optimization using 4-dimensional parameter space.

## Overview

This system implements a 4D extension of the UUPL algorithm to learn user preferences for audio signals. It optimizes four key audio parameters:
- **Amplitude** [30-60]: Controls the overall loudness of the audio signal
- **Frequency** [25-75]: Controls the primary frequency characteristics
- **Density** [10-90]: Controls the density/granularity of the audio texture
- **Gradient** [-50-50]: Controls the envelope gradient characteristics

The system uses Gaussian Process (GP) optimization with uncertainty modeling to efficiently learn user preferences through interactive comparisons.

## System Architecture

```
user_study/
├── audio_preference_interface.py    # Main GUI application
├── gp_4d.py                        # 4D Gaussian Process implementation
├── audio_generator.py              # Audio signal generation and playback
├── README.md                       # This documentation
└── requirements.txt                # Python dependencies (if needed)
```

## Features

### 🎵 Audio Generation
- Uses `generateSignal6params` function for high-quality audio synthesis
- Real-time parameter adjustment and audio generation
- Support for multiple audio playback backends (pygame/sounddevice)
- Automatic audio normalization to prevent clipping

### 🧠 4D UUPL Algorithm
- Extended GP_ours algorithm for 4-dimensional parameter optimization
- Parameter normalization to [0,1] space for efficient optimization
- Information gain maximization for optimal query selection
- Uncertainty-aware preference modeling

### 🖥️ Interactive Interface
- **Interactive Mode**: Manual audio comparison and preference selection
- **Test Mode**: Automatic synthetic preference evaluation for testing
- Real-time visualization of parameter evolution
- Convergence analysis and performance metrics
- Audio waveform and spectrum visualization

### 📊 Visualization & Analysis
- **Parameter Evolution**: Track how each parameter changes over iterations
- **Convergence Analysis**: Information gain, uncertainty levels, parameter space exploration
- **Audio Visualization**: Waveforms and frequency spectra of current audio candidates
- Progress tracking with iteration counts and best parameters display

## Installation

### Prerequisites
```bash
# Required Python packages
pip install numpy scipy matplotlib tkinter threading
pip install pygame  # For audio playback (recommended)
# OR
pip install sounddevice  # Alternative audio backend
```

### Setup
1. Ensure you have the parent UUPL system installed with:
   - `GP_ours.py` - Base Gaussian Process implementation
   - `util.py` - Utility functions
   - `audiosignal/generateSignal6params.py` - Audio generation function

2. Place the `user_study` folder in your UUPL main directory

3. The system will automatically detect available audio backends

## Usage

### Quick Start
```bash
cd user_study
python audio_preference_interface.py
```

### Interactive Mode
1. **Start Learning**: Click "Start Learning" to initialize the system
2. **Generate Audio**: Click "Generate Audio Pair" to create two audio candidates
3. **Listen & Compare**: 
   - Use "▶ Play Audio 1" and "▶ Play Audio 2" to listen to candidates
   - Compare the audio signals and decide which you prefer
4. **Select Preference**: Click "Select Audio 1" or "Select Audio 2"
5. **Set Uncertainty**: Adjust the uncertainty slider (1=Certain, 5=Very Uncertain)
6. **Execute Choice**: Click "Execute Choice" to update the model
7. **Repeat**: Continue the process to refine the learned preferences

### Test Mode
1. Switch to "Test (Auto)" mode before starting
2. Click "Start Learning" to begin automatic evaluation
3. The system will run synthetic preference comparisons automatically
4. Monitor the real-time parameter evolution and convergence

### Interface Controls

| Control | Function |
|---------|----------|
| **Mode Selection** | Switch between Interactive and Test modes |
| **Generate Audio Pair** | Create new audio candidates for comparison |
| **Play Audio 1/2** | Play the generated audio signals |
| **Select Audio 1/2** | Choose preferred audio signal |
| **Uncertainty Slider** | Set confidence level (1-5) |
| **Execute Choice** | Update model with preference |
| **Start Learning** | Initialize the GP optimization |
| **Pause** | Pause automatic test mode |
| **Reset** | Clear all data and restart |

## Technical Details

### 4D Gaussian Process
The system extends the original GP_ours implementation to handle 4-dimensional parameter optimization:

```python
# Parameter normalization
normalized[0] = (amplitude - 30) / (60 - 30)    # Amplitude [30,60] → [0,1]
normalized[1] = (frequency - 25) / (75 - 25)    # Frequency [25,75] → [0,1]
normalized[2] = (density - 10) / (90 - 10)      # Density [10,90] → [0,1]
normalized[3] = (gradient - (-50)) / (50-(-50)) # Gradient [-50,50] → [0,1]
```

### Audio Generation
Audio signals are generated using the `generateSignal6params` function with fixed parameters:
- **Duration**: 4 seconds
- **Cycles**: 1
- **Sampling Rate**: 44.1 kHz

### Uncertainty Modeling
User uncertainty is mapped to confidence weights:
- Level 1 (Certain): 0.1
- Level 2: 0.3
- Level 3 (Neutral): 0.5
- Level 4: 0.8
- Level 5 (Very Uncertain): 1.0

## Visualization Tabs

### 1. Parameter Evolution
- Real-time tracking of all 4 parameters over iterations
- Individual subplots for Amplitude, Frequency, Density, and Gradient
- Clear parameter range visualization

### 2. Convergence Analysis
- **Information Gain**: Measures learning efficiency per iteration
- **Uncertainty Levels**: User confidence over time
- **Parameter Space Exploration**: 2D projection of parameter combinations
- **Cumulative Preferences**: Track positive/negative choices

### 3. Audio Visualization
- **Waveforms**: Time-domain representation of current audio candidates
- **Frequency Spectra**: FFT analysis showing frequency content
- Side-by-side comparison of Audio 1 vs Audio 2

## Performance & Testing

### Synthetic Evaluation (Test Mode)
The system includes a synthetic preference function for automated testing:
- Uses distance-based preference to "ideal" parameters [45, 50, 50, 0]
- Adds Gaussian noise to simulate human variability
- Random uncertainty levels for robust testing

### Expected Performance
- **Convergence**: Typically converges within 20-30 iterations
- **Audio Quality**: High-quality synthesis with normalized output
- **Response Time**: Real-time audio generation and GUI updates
- **Memory Usage**: Efficient storage of preference history

## Troubleshooting

### Common Issues

**Audio Playback Problems**:
```bash
# Install audio backend
pip install pygame
# OR
pip install sounddevice
```

**Import Errors**:
- Ensure parent directory contains `GP_ours.py` and `util.py`
- Check that `audiosignal/generateSignal6params.py` exists

**GUI Issues**:
- Ensure tkinter is properly installed
- On Linux: `sudo apt-get install python3-tk`

**Performance Issues**:
- Reduce visualization update frequency
- Check available memory for large preference histories

## Customization

### Modifying Parameters
Edit parameter ranges in `audio_preference_interface.py`:
```python
self.param_ranges = {
    'amplitude': [30, 60],    # Modify these ranges
    'frequency': [25, 75],
    'density': [10, 90],
    'gradient': [-50, 50]
}
```

### Changing GP Parameters
Adjust Gaussian Process settings:
```python
self.theta = 0.5          # Length scale parameter
self.noise_level = 0.1    # Noise level
self.max_iterations = 50  # Maximum iterations
```

### Audio Settings
Modify audio generation parameters in `audio_generator.py`:
```python
self.duration = 4      # Signal duration (seconds)
self.cycles = 1        # Number of cycles
self.fs = 44100        # Sampling frequency
```

## File Descriptions

### `audio_preference_interface.py`
Main GUI application implementing the complete user interface with:
- Interactive audio comparison system
- Real-time visualization panels
- Parameter tracking and status display
- Test automation capabilities

### `gp_4d.py`
4D extension of the GP_ours algorithm featuring:
- Parameter normalization/denormalization
- 4D RBF kernel implementation
- Optimal query point selection
- Information gain calculation

### `audio_generator.py`
Audio signal generation and playback utilities:
- Integration with `generateSignal6params`
- Multi-backend audio playback support
- Audio feature calculation
- WAV file export capabilities

## Research Applications

This system is designed for research in:
- **Human-Computer Interaction**: Understanding user preferences in audio domain
- **Machine Learning**: Gaussian Process optimization with uncertainty
- **Audio Signal Processing**: Perceptual audio parameter optimization
- **User Experience**: Interactive preference learning interfaces

## Contributing

When extending this system:
1. Maintain the modular architecture
2. Follow the existing coding style
3. Add comprehensive error handling
4. Update documentation for new features
5. Test both interactive and automatic modes

## License

This system is part of the UUPL project and follows the same licensing terms as the parent project.

## Citation

If you use this system in research, please cite the original UUPL work and mention the 4D audio extension:

```bibtex
@software{uupl_4d_audio,
  title={4D Audio Preference Learning System},
  author={UUPL Development Team},
  year={2024},
  note={Extension of UUPL for audio signal optimization}
}
```