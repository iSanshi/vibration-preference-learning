#!/usr/bin/env python3
"""
Audio Preference Learning Interface

Interactive interface for learning user preferences for audio signals using 4D UUPL.
Optimizes amplitude, frequency, density, and gradient parameters.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from datetime import datetime

# Import our modules
from gp_4d import GaussianProcess4D
from audio_generator import AudioGenerator


class AudioPreferenceInterface:
    """Main interface for audio preference learning."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Preference Learning - 4D UUPL")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.setup_parameters()
        self.setup_variables()
        
        # Create GUI
        self.create_widgets()
        self.setup_plots()
        
        # Initialize audio generator and GP
        self.audio_gen = AudioGenerator()
        self.gp = None
        
    def setup_parameters(self):
        """Initialize system parameters."""
        # Parameter ranges (physical units)
        self.param_ranges = {
            'amplitude': [30, 60],
            'frequency': [25, 75],
            'density': [10, 90],
            'gradient': [-50, 50]
        }
        
        # GP parameters
        self.initial_point = [0.5, 0.5, 0.5, 0.5]  # Normalized center
        self.theta = 0.5
        self.noise_level = 0.1
        
        # Uncertainty levels
        self.uncertainty_dict = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
        
    def setup_variables(self):
        """Initialize control variables."""
        # Simulation state
        self.current_iteration = 0
        self.max_iterations = 50
        self.is_running = False
        self.is_test_mode = False
        
        # Data storage
        self.preference_history = []
        self.parameter_history = []
        self.uncertainty_history = []
        self.info_gain_history = []
        
        # Current query state
        self.current_candidates = None
        self.current_audio_data = {}  # Store generated audio
        self.selected_choice = None
        
        # Preference dictionary for GP
        self.pref_dict = {}
        
    def create_widgets(self):
        """Create the GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control panel
        self.create_control_panel(main_frame)
        
        # Create visualization area
        self.create_visualization_area(main_frame)
        
    def create_control_panel(self, parent):
        """Create the control panel."""
        control_frame = ttk.LabelFrame(parent, text="Audio Preference Learning Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="Interactive")
        ttk.Radiobutton(mode_frame, text="Interactive", variable=self.mode_var, 
                       value="Interactive", command=self.on_mode_change).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Radiobutton(mode_frame, text="Test (Auto)", variable=self.mode_var, 
                       value="Test", command=self.on_mode_change).pack(side=tk.LEFT)
        
        # Interactive controls
        self.interactive_frame = ttk.LabelFrame(control_frame, text="Audio Comparison", padding=5)
        self.interactive_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current candidates display
        self.candidates_var = tk.StringVar(value="Click 'Generate' to start learning")
        candidates_label = ttk.Label(self.interactive_frame, textvariable=self.candidates_var, 
                                    wraplength=600, font=("Arial", 9))
        candidates_label.pack(pady=(0, 10))
        
        # Audio control buttons
        audio_frame = ttk.Frame(self.interactive_frame)
        audio_frame.pack(pady=(0, 10))
        
        # Audio 1 controls
        audio1_frame = ttk.LabelFrame(audio_frame, text="Audio 1", padding=5)
        audio1_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.play1_btn = ttk.Button(audio1_frame, text="▶ Play Audio 1", 
                                   command=lambda: self.play_audio(1), state=tk.DISABLED)
        self.play1_btn.pack(pady=(0, 5))
        
        self.select1_btn = ttk.Button(audio1_frame, text="Select Audio 1", 
                                     command=lambda: self.select_audio(1), state=tk.DISABLED)
        self.select1_btn.pack()
        
        # Audio 2 controls
        audio2_frame = ttk.LabelFrame(audio_frame, text="Audio 2", padding=5)
        audio2_frame.pack(side=tk.LEFT)
        
        self.play2_btn = ttk.Button(audio2_frame, text="▶ Play Audio 2", 
                                   command=lambda: self.play_audio(2), state=tk.DISABLED)
        self.play2_btn.pack(pady=(0, 5))
        
        self.select2_btn = ttk.Button(audio2_frame, text="Select Audio 2", 
                                     command=lambda: self.select_audio(2), state=tk.DISABLED)
        self.select2_btn.pack()
        
        # Uncertainty and execution controls
        control_row = ttk.Frame(self.interactive_frame)
        control_row.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(control_row, text="Uncertainty Level:").pack(side=tk.LEFT)
        self.uncertainty_var = tk.IntVar(value=1)
        uncertainty_scale = ttk.Scale(control_row, from_=1, to=5, orient=tk.HORIZONTAL, 
                                     variable=self.uncertainty_var, length=120)
        uncertainty_scale.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(control_row, text="1=Certain ... 5=Very Uncertain").pack(side=tk.LEFT, padx=(0, 20))
        
        self.generate_btn = ttk.Button(control_row, text="Generate Audio Pair", 
                                      command=self.generate_audio_candidates, state=tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.execute_btn = ttk.Button(control_row, text="Execute Choice", 
                                     command=self.execute_choice, state=tk.DISABLED)
        self.execute_btn.pack(side=tk.LEFT)
        
        # Main controls
        main_controls = ttk.Frame(control_frame)
        main_controls.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(main_controls, text="Start Learning", 
                                   command=self.start_learning)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_btn = ttk.Button(main_controls, text="Pause", 
                                   command=self.pause_learning, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_btn = ttk.Button(main_controls, text="Reset", 
                                   command=self.reset_learning)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Status display
        status_frame = ttk.Frame(main_controls)
        status_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(status_frame, text="Iteration:").pack(side=tk.LEFT)
        self.iteration_var = tk.StringVar(value="0")
        ttk.Label(status_frame, textvariable=self.iteration_var, 
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(status_frame, text="Best Parameters:").pack(side=tk.LEFT)
        self.best_params_var = tk.StringVar(value="N/A")
        ttk.Label(status_frame, textvariable=self.best_params_var, 
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_controls, variable=self.progress_var, 
                                          maximum=100, length=200)
        self.progress_bar.pack(side=tk.RIGHT)
        
    def create_visualization_area(self, parent):
        """Create the visualization area."""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Parameter evolution tab
        self.params_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.params_frame, text="Parameter Evolution")
        
        # Convergence analysis tab
        self.convergence_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.convergence_frame, text="Convergence Analysis")
        
        # Audio visualization tab
        self.audio_viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.audio_viz_frame, text="Audio Visualization")
        
    def setup_plots(self):
        """Setup matplotlib plots."""
        # Parameter evolution plot
        self.params_fig = Figure(figsize=(12, 8))
        self.params_canvas = FigureCanvasTkAgg(self.params_fig, self.params_frame)
        self.params_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Convergence plot
        self.conv_fig = Figure(figsize=(12, 8))
        self.conv_canvas = FigureCanvasTkAgg(self.conv_fig, self.convergence_frame)
        self.conv_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Audio visualization plot
        self.audio_fig = Figure(figsize=(12, 8))
        self.audio_canvas = FigureCanvasTkAgg(self.audio_fig, self.audio_viz_frame)
        self.audio_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.update_parameter_plot()
        self.update_convergence_plot()
        self.update_audio_plot()
        
    def on_mode_change(self):
        """Handle mode change between Interactive and Test."""
        self.is_test_mode = (self.mode_var.get() == "Test")
        
        if self.is_test_mode:
            # Hide interactive controls for test mode
            self.interactive_frame.pack_forget()
        else:
            # Show interactive controls
            self.interactive_frame.pack(fill=tk.X, pady=(0, 10))
            
    def start_learning(self):
        """Start the preference learning process."""
        # Initialize GP
        self.gp = GaussianProcess4D(self.initial_point, self.theta, self.noise_level)
        
        # Initialize with some dummy preferences to start
        center_point = [0.5, 0.5, 0.5, 0.5]
        self.pref_dict = {tuple(center_point): 1.0}
        self.gp.updateParameters([center_point, center_point], 0, 1, self.pref_dict)
        
        if self.is_test_mode:
            # Start automatic test mode
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            
            # Start test in separate thread
            self.test_thread = threading.Thread(target=self.run_test_mode)
            self.test_thread.daemon = True
            self.test_thread.start()
        else:
            # Enable interactive controls
            self.generate_btn.config(state=tk.NORMAL)
            self.start_btn.config(text="Initialize", state=tk.DISABLED)
            messagebox.showinfo("Interactive Mode", 
                               "Click 'Generate Audio Pair' to start learning!")
            
    def generate_audio_candidates(self):
        """Generate a pair of audio candidates for comparison."""
        if not self.gp:
            messagebox.showerror("Error", "Please start learning first!")
            return
            
        try:
            # Find optimal query points
            optimal_query, info_gain = self.gp.find_optimal_query_4d()
            
            # Extract the two 4D points
            point1_norm = optimal_query[:4]
            point2_norm = optimal_query[4:]
            
            # Convert to physical parameters
            point1_phys = self.gp.denormalize_parameters(point1_norm)
            point2_phys = self.gp.denormalize_parameters(point2_norm)
            
            # Generate audio signals
            time1, audio1, meta1 = self.audio_gen.generate_signal(*point1_phys)
            time2, audio2, meta2 = self.audio_gen.generate_signal(*point2_phys)
            
            # Store current candidates
            self.current_candidates = {
                'points_norm': [point1_norm, point2_norm],
                'points_phys': [point1_phys, point2_phys],
                'info_gain': info_gain
            }
            
            self.current_audio_data = {
                1: {'time': time1, 'data': audio1, 'meta': meta1},
                2: {'time': time2, 'data': audio2, 'meta': meta2}
            }
            
            # Update display
            self.candidates_var.set(
                f"Audio 1: A={point1_phys[0]:.1f}, F={point1_phys[1]:.1f}, D={point1_phys[2]:.1f}, G={point1_phys[3]:.1f} | "
                f"Audio 2: A={point2_phys[0]:.1f}, F={point2_phys[1]:.1f}, D={point2_phys[2]:.1f}, G={point2_phys[3]:.1f}"
            )
            
            # Enable audio controls
            self.play1_btn.config(state=tk.NORMAL)
            self.play2_btn.config(state=tk.NORMAL)
            self.select1_btn.config(state=tk.NORMAL)
            self.select2_btn.config(state=tk.NORMAL)
            self.generate_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate candidates: {e}")
            
    def play_audio(self, audio_num):
        """Play the specified audio."""
        if audio_num not in self.current_audio_data:
            messagebox.showerror("Error", "No audio data available!")
            return
            
        try:
            audio_data = self.current_audio_data[audio_num]['data']
            
            # Stop any currently playing audio
            self.audio_gen.stop_audio()
            
            # Play the selected audio (non-blocking)
            success = self.audio_gen.play_audio(audio_data, blocking=False)
            
            if not success:
                messagebox.showwarning("Warning", "Audio playback failed. Check audio system.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Audio playback error: {e}")
            
    def select_audio(self, audio_num):
        """Select preferred audio."""
        self.selected_choice = audio_num
        
        # Update button states
        if audio_num == 1:
            self.select1_btn.config(text="✓ Audio 1 Selected", state=tk.DISABLED)
            self.select2_btn.config(text="Select Audio 2", state=tk.NORMAL)
        else:
            self.select2_btn.config(text="✓ Audio 2 Selected", state=tk.DISABLED)
            self.select1_btn.config(text="Select Audio 1", state=tk.NORMAL)
            
        # Enable execute button
        self.execute_btn.config(state=tk.NORMAL)
        
    def execute_choice(self):
        """Execute the user's choice and update the model."""
        if not self.selected_choice or not self.current_candidates:
            messagebox.showerror("Error", "Please select an audio first!")
            return
            
        try:
            # Get choice and uncertainty
            user_choice = 1 if self.selected_choice == 1 else -1
            uncertainty_level = self.uncertainty_var.get()
            
            # Get the candidate points
            point1_norm, point2_norm = self.current_candidates['points_norm']
            
            # Update preference dictionary
            key1 = tuple(point1_norm)
            key2 = tuple(point2_norm)
            
            self.pref_dict[key1] = self.pref_dict.get(key1, 0) + self.uncertainty_dict[uncertainty_level]
            self.pref_dict[key2] = self.pref_dict.get(key2, 0) + self.uncertainty_dict[uncertainty_level]
            
            # Update GP
            self.gp.updateParameters(
                [point1_norm, point2_norm],
                user_choice,
                uncertainty_level,
                self.pref_dict
            )
            
            # Store data
            selected_point = point1_norm if self.selected_choice == 1 else point2_norm
            selected_phys = self.current_candidates['points_phys'][self.selected_choice - 1]
            
            self.preference_history.append(user_choice)
            self.parameter_history.append(selected_phys)
            self.uncertainty_history.append(uncertainty_level)
            self.info_gain_history.append(self.current_candidates['info_gain'])
            
            self.current_iteration += 1
            
            # Update displays
            self.update_status()
            self.update_plots()
            
            # Reset for next iteration
            self.reset_interactive_controls()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute choice: {e}")
            
    def reset_interactive_controls(self):
        """Reset interactive controls for next iteration."""
        self.current_candidates = None
        self.current_audio_data = {}
        self.selected_choice = None
        
        self.candidates_var.set("Click 'Generate Audio Pair' to continue learning")
        
        self.play1_btn.config(state=tk.DISABLED)
        self.play2_btn.config(state=tk.DISABLED)
        self.select1_btn.config(text="Select Audio 1", state=tk.DISABLED)
        self.select2_btn.config(text="Select Audio 2", state=tk.DISABLED)
        self.execute_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        
    def run_test_mode(self):
        """Run automatic test mode."""
        while self.is_running and self.current_iteration < self.max_iterations:
            try:
                # Generate candidates automatically
                optimal_query, info_gain = self.gp.find_optimal_query_4d()
                
                point1_norm = optimal_query[:4]
                point2_norm = optimal_query[4:]
                point1_phys = self.gp.denormalize_parameters(point1_norm)
                point2_phys = self.gp.denormalize_parameters(point2_norm)
                
                # Generate audio and evaluate using a synthetic preference function
                _, audio1, _ = self.audio_gen.generate_signal(*point1_phys)
                _, audio2, _ = self.audio_gen.generate_signal(*point2_phys)
                
                # Synthetic preference based on parameter distance from "ideal"
                ideal_params = [45, 50, 50, 0]  # Middle values
                dist1 = np.linalg.norm(np.array(point1_phys) - np.array(ideal_params))
                dist2 = np.linalg.norm(np.array(point2_phys) - np.array(ideal_params))
                
                # Prefer the point closer to ideal (with some noise)
                noise = np.random.normal(0, 0.1)
                user_choice = 1 if (dist1 + noise) < dist2 else -1
                
                # Random uncertainty level
                uncertainty_level = np.random.randint(1, 4)
                
                # Update model
                key1 = tuple(point1_norm)
                key2 = tuple(point2_norm)
                
                self.pref_dict[key1] = self.pref_dict.get(key1, 0) + self.uncertainty_dict[uncertainty_level]
                self.pref_dict[key2] = self.pref_dict.get(key2, 0) + self.uncertainty_dict[uncertainty_level]
                
                self.gp.updateParameters(
                    [point1_norm, point2_norm],
                    user_choice,
                    uncertainty_level,
                    self.pref_dict
                )
                
                # Store data
                selected_point = point1_norm if user_choice == 1 else point2_norm
                selected_phys = point1_phys if user_choice == 1 else point2_phys
                
                self.preference_history.append(user_choice)
                self.parameter_history.append(selected_phys)
                self.uncertainty_history.append(uncertainty_level)
                self.info_gain_history.append(info_gain)
                
                self.current_iteration += 1
                
                # Update GUI
                self.root.after(0, self.update_status)
                self.root.after(0, self.update_plots)
                
                time.sleep(0.5)  # Control speed
                
            except Exception as e:
                print(f"Error in test mode: {e}")
                break
                
        # Test finished
        self.root.after(0, self.test_finished)
        
    def test_finished(self):
        """Handle test mode completion."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Test Complete", 
                           f"Test completed after {self.current_iteration} iterations!")
        
    def pause_learning(self):
        """Pause the learning process."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
    def reset_learning(self):
        """Reset the learning process."""
        self.is_running = False
        self.current_iteration = 0
        
        # Clear data
        self.preference_history = []
        self.parameter_history = []
        self.uncertainty_history = []
        self.info_gain_history = []
        self.pref_dict = {}
        
        # Reset GP
        self.gp = None
        
        # Reset controls
        self.reset_interactive_controls()
        self.start_btn.config(text="Start Learning", state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.DISABLED)
        
        # Update displays
        self.update_status()
        self.update_plots()
        
    def update_status(self):
        """Update status displays."""
        self.iteration_var.set(str(self.current_iteration))
        
        # Show best parameters so far
        if self.parameter_history:
            best_params = self.parameter_history[-1]  # Most recent
            self.best_params_var.set(
                f"A={best_params[0]:.1f}, F={best_params[1]:.1f}, D={best_params[2]:.1f}, G={best_params[3]:.1f}"
            )
        
        # Update progress
        progress = (self.current_iteration / self.max_iterations) * 100
        self.progress_var.set(progress)
        
    def update_plots(self):
        """Update all plots."""
        self.update_parameter_plot()
        self.update_convergence_plot()
        self.update_audio_plot()
        
    def update_parameter_plot(self):
        """Update parameter evolution plot."""
        self.params_fig.clear()
        
        if not self.parameter_history:
            ax = self.params_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data yet\nStart learning to see parameter evolution', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.params_canvas.draw()
            return
            
        params_array = np.array(self.parameter_history)
        iterations = range(1, len(params_array) + 1)
        
        # Plot parameter evolution
        ax1 = self.params_fig.add_subplot(221)
        ax1.plot(iterations, params_array[:, 0], 'b-o', markersize=4)
        ax1.set_title('Amplitude Evolution')
        ax1.set_ylabel('Amplitude [30-60]')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(25, 65)
        
        ax2 = self.params_fig.add_subplot(222)
        ax2.plot(iterations, params_array[:, 1], 'g-o', markersize=4)
        ax2.set_title('Frequency Evolution')
        ax2.set_ylabel('Frequency [25-75]')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(20, 80)
        
        ax3 = self.params_fig.add_subplot(223)
        ax3.plot(iterations, params_array[:, 2], 'r-o', markersize=4)
        ax3.set_title('Density Evolution')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Density [10-90]')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(5, 95)
        
        ax4 = self.params_fig.add_subplot(224)
        ax4.plot(iterations, params_array[:, 3], 'm-o', markersize=4)
        ax4.set_title('Gradient Evolution')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Gradient [-50-50]')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-55, 55)
        
        self.params_fig.tight_layout()
        self.params_canvas.draw()
        
    def update_convergence_plot(self):
        """Update convergence analysis plot."""
        self.conv_fig.clear()
        
        if not self.info_gain_history:
            ax = self.conv_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No convergence data yet\nStart learning to see analysis', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.conv_canvas.draw()
            return
            
        iterations = range(1, len(self.info_gain_history) + 1)
        
        # Information gain plot
        ax1 = self.conv_fig.add_subplot(221)
        ax1.plot(iterations, self.info_gain_history, 'b-', linewidth=2)
        ax1.set_title('Information Gain per Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Information Gain')
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty levels
        ax2 = self.conv_fig.add_subplot(222)
        if self.uncertainty_history:
            ax2.plot(iterations, self.uncertainty_history, 'g-o', linewidth=2, markersize=4)
            ax2.set_title('Uncertainty Levels')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Uncertainty Level')
            ax2.set_ylim(0.5, 5.5)
            ax2.grid(True, alpha=0.3)
        
        # Parameter space exploration (2D projection)
        ax3 = self.conv_fig.add_subplot(223)
        if self.parameter_history:
            params = np.array(self.parameter_history)
            scatter = ax3.scatter(params[:, 0], params[:, 1], 
                                c=range(len(params)), cmap='viridis', s=50, alpha=0.7)
            ax3.set_title('Parameter Space Exploration (Amp vs Freq)')
            ax3.set_xlabel('Amplitude')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
        # Cumulative preferences
        ax4 = self.conv_fig.add_subplot(224)
        if self.preference_history:
            cumulative_pos = np.cumsum([1 if p > 0 else 0 for p in self.preference_history])
            ax4.plot(iterations, cumulative_pos, 'r-', linewidth=2)
            ax4.set_title('Cumulative Positive Choices')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Count')
            ax4.grid(True, alpha=0.3)
        
        self.conv_fig.tight_layout()
        self.conv_canvas.draw()
        
    def update_audio_plot(self):
        """Update audio visualization plot."""
        self.audio_fig.clear()
        
        if not self.current_audio_data:
            ax = self.audio_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No audio data\nGenerate audio pair to see waveforms', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.audio_canvas.draw()
            return
            
        # Plot current audio waveforms
        if 1 in self.current_audio_data and 2 in self.current_audio_data:
            
            # Audio 1 waveform
            ax1 = self.audio_fig.add_subplot(221)
            time1 = self.current_audio_data[1]['time']
            data1 = self.current_audio_data[1]['data']
            ax1.plot(time1, data1, 'b-', linewidth=1)
            ax1.set_title('Audio 1 Waveform')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            
            # Audio 2 waveform
            ax2 = self.audio_fig.add_subplot(222)
            time2 = self.current_audio_data[2]['time']
            data2 = self.current_audio_data[2]['data']
            ax2.plot(time2, data2, 'r-', linewidth=1)
            ax2.set_title('Audio 2 Waveform')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
            # Spectrograms or FFT
            ax3 = self.audio_fig.add_subplot(223)
            fft1 = np.fft.fft(data1)
            freqs1 = np.fft.fftfreq(len(data1), 1/self.audio_gen.fs)
            ax3.plot(freqs1[:len(freqs1)//2], np.abs(fft1[:len(fft1)//2]), 'b-')
            ax3.set_title('Audio 1 Spectrum')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Magnitude')
            ax3.grid(True, alpha=0.3)
            
            ax4 = self.audio_fig.add_subplot(224)
            fft2 = np.fft.fft(data2)
            freqs2 = np.fft.fftfreq(len(data2), 1/self.audio_gen.fs)
            ax4.plot(freqs2[:len(freqs2)//2], np.abs(fft2[:len(fft2)//2]), 'r-')
            ax4.set_title('Audio 2 Spectrum')
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Magnitude')
            ax4.grid(True, alpha=0.3)
            
        self.audio_fig.tight_layout()
        self.audio_canvas.draw()


def main():
    """Main function to run the audio preference interface."""
    print("Audio Preference Learning Interface")
    print("==" * 20)
    
    try:
        root = tk.Tk()
        app = AudioPreferenceInterface(root)
        root.mainloop()
    except Exception as e:
        print(f"Error launching interface: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())