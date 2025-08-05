#!/usr/bin/env python3
"""
UUPL Interactive Interface

This module provides an interactive visualization interface for UUPL (Uncertainty-aware User Preference Learning)
simulation with real-time updates, convergence analysis, and user interaction controls.

Usage:
    python interactive_interface.py

Dependencies are checked automatically on startup.
"""

import sys
import subprocess
import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox
# import matplotlib.pyplot as plt  # Not directly used, imported via Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from scipy.stats import multivariate_normal
from GP_ours import GaussianProcess
from util import *
import scipy.optimize as opt


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == 'tkinter':
            try:
                import tkinter as _  # Import and discard to check availability
            except ImportError:
                missing_packages.append('tkinter')
        else:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(package)
    
    return missing_packages


def install_missing_packages(packages):
    """Install missing packages using pip."""
    if packages:
        print(f"Missing packages detected: {', '.join(packages)}")
        print("Installing missing packages...")
        
        for package in packages:
            if package == 'tkinter':
                print("tkinter is a built-in Python package. Please ensure you have a complete Python installation.")
                continue
            
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install it manually.")
                return False
    return True


class UUPLInteractiveInterface:
    """Interactive interface for UUPL simulation with real-time updates."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("UUPL Interactive Simulation Interface")
        self.root.geometry("1400x800")
        
        # Initialize parameters
        self.setup_parameters()
        
        # Simulation control
        self.is_running = False
        self.simulation_thread = None
        self.current_iteration = 0
        self.max_iterations = 50
        
        # Create GUI components
        self.create_widgets()
        
        # Initialize plots
        self.setup_plots()
        
    def setup_parameters(self):
        """Initialize simulation parameters."""
        # GMM parameters
        self.means = [np.array([-2, 3]), np.array([0, -3]), np.array([2, 2])]
        self.covariances = [
            np.array([[2, 1], [1, 2]]),
            np.array([[10, -3], [-3, 4]]),
            np.array([[2, 0], [0, 2]])
        ]
        self.weights = [5/1.6, 22/1.6, 10/1.6]
        
        # GP parameters
        self.initial_point = [1, 1]
        self.theta = 0.5
        self.noise_level = 0.1
        
        # Uncertainty parameters
        self.uncertainty_dict = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
        self.uncertainty_thresh = [0.01, 0.05, 0.1, 0.2]
        
        # Data storage
        self.correlation_history = []
        self.query_history = []
        self.info_gain_history = []
        self.uncertainty_history = []
        
        # Manual choice control
        self.waiting_for_choice = False
        self.user_choice = None
        self.current_query_pair = None
        
        # Manual simulation control
        self.manual_mode = False
        self.current_gp = None
        self.current_pref_dict = {}
        
        # Interactive step control
        self.current_query_candidates = None
        self.selected_point = None
        self.waiting_for_point_selection = False
        
    def create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Method selection
        method_frame = ttk.Frame(control_frame)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(method_frame, text="Control Mode:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="Interactive")
        method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, 
                                   values=["Interactive", "Automatic"], 
                                   state="readonly", width=15)
        method_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        # Add explanation label
        explanation_frame = ttk.Frame(method_frame)
        explanation_frame.pack(side=tk.LEFT, padx=(10, 0))
        self.mode_explanation = tk.StringVar(value="Interactive: Manual step-by-step control")
        ttk.Label(explanation_frame, textvariable=self.mode_explanation, font=("Arial", 8), 
                 foreground="gray").pack(side=tk.LEFT)
        
        # Bind mode change to update explanation
        method_combo.bind("<<ComboboxSelected>>", self.update_mode_explanation)
        
        # User choice buttons
        choice_frame = ttk.LabelFrame(control_frame, text="User Choice Mode", padding=5)
        choice_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.choice_mode = tk.StringVar(value="Auto")
        
        # Auto mode
        auto_frame = ttk.Frame(choice_frame)
        auto_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Radiobutton(auto_frame, text="Auto Mode", variable=self.choice_mode, 
                       value="Auto").pack(side=tk.LEFT)
        
        # Manual mode with buttons
        manual_frame = ttk.Frame(choice_frame)
        manual_frame.pack(fill=tk.X)
        ttk.Radiobutton(manual_frame, text="Manual Mode", variable=self.choice_mode, 
                       value="Manual").pack(side=tk.LEFT)
        
        # Choice buttons (initially disabled)
        self.choice_buttons_frame = ttk.Frame(manual_frame)
        self.choice_buttons_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        self.choice1_btn = ttk.Button(self.choice_buttons_frame, text="Choice 1", 
                                     command=lambda: self.make_choice(1), state=tk.DISABLED)
        self.choice1_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.choice2_btn = ttk.Button(self.choice_buttons_frame, text="Choice 2", 
                                     command=lambda: self.make_choice(2), state=tk.DISABLED)
        self.choice2_btn.pack(side=tk.LEFT)
        
        # Current choice display
        self.current_choice_var = tk.StringVar(value="Waiting...")
        self.choice_label = ttk.Label(self.choice_buttons_frame, textvariable=self.current_choice_var)
        self.choice_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Interactive step control frame
        step_frame = ttk.LabelFrame(control_frame, text="Interactive Step Control", padding=5)
        step_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current query points display
        query_display_frame = ttk.Frame(step_frame)
        query_display_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(query_display_frame, text="Current Query Candidates:").pack(side=tk.LEFT)
        self.query_candidates_var = tk.StringVar(value="Not generated yet")
        self.query_candidates_label = ttk.Label(query_display_frame, textvariable=self.query_candidates_var, 
                                               font=("Arial", 9))
        self.query_candidates_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Point selection buttons
        selection_frame = ttk.Frame(step_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.select_point1_btn = ttk.Button(selection_frame, text="Select Point 1", 
                                           command=lambda: self.select_query_point(1), state=tk.DISABLED)
        self.select_point1_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.select_point2_btn = ttk.Button(selection_frame, text="Select Point 2", 
                                           command=lambda: self.select_query_point(2), state=tk.DISABLED)
        self.select_point2_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Uncertainty input
        uncertainty_input_frame = ttk.Frame(step_frame)
        uncertainty_input_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(uncertainty_input_frame, text="Uncertainty Level:").pack(side=tk.LEFT)
        self.manual_uncertainty_var = tk.IntVar(value=1)
        uncertainty_scale = tk.Scale(uncertainty_input_frame, from_=1, to=5, 
                                   orient=tk.HORIZONTAL, variable=self.manual_uncertainty_var, length=100)
        uncertainty_scale.pack(side=tk.LEFT, padx=(5, 20))
        
        # Manual step control
        manual_control_frame = ttk.Frame(step_frame)
        manual_control_frame.pack(fill=tk.X)
        
        self.generate_query_btn = ttk.Button(manual_control_frame, text="Generate Query Points", 
                                           command=self.generate_query_points, state=tk.DISABLED)
        self.generate_query_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.next_step_btn = ttk.Button(manual_control_frame, text="Next Step", 
                                       command=self.next_interactive_step, state=tk.DISABLED)
        self.next_step_btn.pack(side=tk.LEFT)
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(speed_frame, from_=0.1, to=5.0, resolution=0.1,
                              orient=tk.HORIZONTAL, variable=self.speed_var, length=100)
        speed_scale.pack(side=tk.LEFT, padx=(5, 20))
        
        # Current status display
        status_frame = ttk.LabelFrame(control_frame, text="Current Status", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Uncertainty display
        uncertainty_frame = ttk.Frame(status_frame)
        uncertainty_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(uncertainty_frame, text="Uncertainty Level:").pack(side=tk.LEFT)
        self.uncertainty_var = tk.StringVar(value="N/A")
        self.uncertainty_label = ttk.Label(uncertainty_frame, textvariable=self.uncertainty_var, 
                                          font=("Arial", 10, "bold"))
        self.uncertainty_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Query display
        query_frame = ttk.Frame(status_frame)
        query_frame.pack(fill=tk.X)
        ttk.Label(query_frame, text="Current Query:").pack(side=tk.LEFT)
        self.query_var = tk.StringVar(value="N/A")
        self.query_label = ttk.Label(query_frame, textvariable=self.query_var)
        self.query_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(button_frame, text="Start Simulation", 
                                   command=self.start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_btn = ttk.Button(button_frame, text="Pause", 
                                   command=self.pause_simulation, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_btn = ttk.Button(button_frame, text="Reset", 
                                   command=self.reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var,
                                           maximum=100, length=200)
        self.progress_bar.pack(side=tk.RIGHT)
        
        # Plot area
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Function surface
        self.surface_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.surface_frame, text="Function Surface")
        
        # Tab 2: Convergence curves
        self.convergence_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.convergence_frame, text="Convergence Analysis")
        
        # Tab 3: Query visualization
        self.query_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.query_frame, text="Query History")
        
    def setup_plots(self):
        """Setup matplotlib plots."""
        # Surface plot
        self.surface_fig = Figure(figsize=(12, 8))
        self.surface_canvas = FigureCanvasTkAgg(self.surface_fig, self.surface_frame)
        self.surface_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Convergence plot
        self.conv_fig = Figure(figsize=(12, 8))
        self.conv_canvas = FigureCanvasTkAgg(self.conv_fig, self.convergence_frame)
        self.conv_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Query plot
        self.query_fig = Figure(figsize=(12, 8))
        self.query_canvas = FigureCanvasTkAgg(self.query_fig, self.query_frame)
        self.query_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.create_gmm_surface()
        self.update_convergence_plot()
        self.update_query_plot()
        
    def create_gmm_surface(self):
        """Create and display the GMM surface."""
        step_size = 0.1
        x, y = np.mgrid[-5:5+step_size:step_size, -5:5+step_size:step_size]
        pos = np.dstack((x, y))
        
        # Calculate GMM PDF
        gmm_pdf = np.zeros(x.shape)
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            rv = multivariate_normal(mean, cov)
            gmm_pdf += weight * rv.pdf(pos)
        
        self.x_grid, self.y_grid, self.gmm_pdf = x, y, gmm_pdf
        
        # Plot original GMM surface
        self.surface_fig.clear()
        ax1 = self.surface_fig.add_subplot(121, projection='3d')
        ax1.plot_surface(x, y, gmm_pdf, cmap='viridis', alpha=0.8)
        ax1.set_title('Original GMM Surface')
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Reward')
        ax1.view_init(elev=30, azim=45)
        
        # Placeholder for learned surface
        ax2 = self.surface_fig.add_subplot(122, projection='3d')
        ax2.set_title('Learned Surface')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Predicted Reward')
        ax2.view_init(elev=30, azim=45)
        
        self.surface_fig.tight_layout()
        self.surface_canvas.draw()
        
    def update_surface_plot(self, gp):
        """Update the learned surface plot."""
        pos = np.dstack((self.x_grid, self.y_grid))
        x_pred = pos.reshape(-1, 2)
        y_pred = gp.mean1pt(x_pred, eval=True)
        y_pred = y_pred.reshape(self.x_grid.shape)
        
        # Update the second subplot
        ax2 = self.surface_fig.axes[1]
        ax2.clear()
        ax2.plot_surface(self.x_grid, self.y_grid, y_pred, cmap='plasma', alpha=0.8)
        ax2.set_title(f'Learned Surface (Iteration {self.current_iteration})')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Predicted Reward')
        ax2.view_init(elev=30, azim=45)
        
        # Add query points
        if self.query_history:
            queries = np.array(self.query_history)
            ax2.scatter(queries[:, 0], queries[:, 1], 
                       [gp.mean1pt(q.reshape(1, -1), eval=True) for q in queries],
                       c='red', s=50, alpha=0.8)
        
        self.surface_canvas.draw()
        
    def update_convergence_plot(self):
        """Update convergence curves."""
        self.conv_fig.clear()
        
        # Correlation plot
        ax1 = self.conv_fig.add_subplot(221)
        if self.correlation_history:
            ax1.plot(self.correlation_history, 'b-', linewidth=2)
            ax1.set_title('Correlation with True Function')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Correlation')
            ax1.grid(True, alpha=0.3)
        
        # Information gain plot
        ax2 = self.conv_fig.add_subplot(222)
        if self.info_gain_history:
            ax2.plot(self.info_gain_history, 'g-', linewidth=2)
            ax2.set_title('Information Gain')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Info Gain')
            ax2.grid(True, alpha=0.3)
        
        # Uncertainty level plot
        ax4 = self.conv_fig.add_subplot(224)
        if self.uncertainty_history:
            ax4.plot(self.uncertainty_history, 'r-', linewidth=2, marker='o', markersize=4)
            ax4.set_title('Uncertainty Level')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Uncertainty Level')
            ax4.set_ylim(0.5, 5.5)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.1, 0.8, f'Iteration: {self.current_iteration}', fontsize=12)
            ax4.text(0.1, 0.6, f'Total Queries: {len(self.query_history)}', fontsize=12)
            if self.correlation_history:
                ax4.text(0.1, 0.4, f'Current Correlation: {self.correlation_history[-1]:.3f}', fontsize=12)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Statistics')
            ax4.axis('off')
        
        # Query distribution
        ax3 = self.conv_fig.add_subplot(223)
        if self.query_history:
            queries = np.array(self.query_history)
            ax3.scatter(queries[:, 0], queries[:, 1], 
                       c=range(len(queries)), cmap='viridis', s=50, alpha=0.7)
            ax3.set_title('Query Distribution')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.grid(True, alpha=0.3)
        
        self.conv_fig.tight_layout()
        self.conv_canvas.draw()
        
    def update_query_plot(self):
        """Update query visualization."""
        self.query_fig.clear()
        
        if not self.query_history:
            ax = self.query_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No queries yet', ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.query_canvas.draw()
            return
        
        # Query trajectory
        ax1 = self.query_fig.add_subplot(121)
        queries = np.array(self.query_history)
        
        # Create contour plot of true function
        x, y = np.mgrid[-5:5:0.2, -5:5:0.2]
        pos = np.dstack((x, y))
        gmm_pdf = np.zeros(x.shape)
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            rv = multivariate_normal(mean, cov)
            gmm_pdf += weight * rv.pdf(pos)
        
        contour = ax1.contour(x, y, gmm_pdf, levels=10, alpha=0.5)
        ax1.clabel(contour, inline=True, fontsize=8)
        
        # Plot query trajectory
        if len(queries) > 1:
            ax1.plot(queries[:, 0], queries[:, 1], 'ro-', markersize=5, alpha=0.7)
        ax1.scatter(queries[-1, 0], queries[-1, 1], c='red', s=100, marker='*', 
                   label='Latest Query')
        
        ax1.set_title('Query Trajectory on True Function')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Query details
        ax2 = self.query_fig.add_subplot(122)
        recent_queries = queries[-10:] if len(queries) > 10 else queries
        
        for i, query in enumerate(recent_queries):
            ax2.text(0.1, 0.9 - i*0.08, f'Query {len(queries)-len(recent_queries)+i+1}: ({query[0]:.2f}, {query[1]:.2f})', 
                    fontsize=10)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Recent Queries')
        ax2.axis('off')
        
        self.query_fig.tight_layout()
        self.query_canvas.draw()
        
    def update_mode_explanation(self, event=None):
        """Update the explanation text based on selected mode."""
        mode = self.method_var.get()
        explanations = {
            "Interactive": "Interactive: Manual step-by-step control with point selection",
            "Automatic": "Automatic: Continuous simulation with optimal query selection"
        }
        self.mode_explanation.set(explanations.get(mode, ""))
        
    def make_choice(self, choice):
        """Handle user manual choice."""
        if self.waiting_for_choice:
            self.user_choice = 1 if choice == 1 else -1
            self.waiting_for_choice = False
            self.choice1_btn.config(state=tk.DISABLED)
            self.choice2_btn.config(state=tk.DISABLED)
            self.current_choice_var.set(f"Choice {choice} selected")
        
    def wait_for_user_choice(self, query1, query2, value1, value2):
        """Wait for user choice in manual mode."""
        if self.choice_mode.get() == "Manual":
            self.waiting_for_choice = True
            self.user_choice = None
            self.current_query_pair = (query1, query2)
            
            # Update button labels with query info
            self.choice1_btn.config(text=f"Choice 1: ({query1[0]:.1f}, {query1[1]:.1f})\nValue: {value1:.3f}", 
                                   state=tk.NORMAL)
            self.choice2_btn.config(text=f"Choice 2: ({query2[0]:.1f}, {query2[1]:.1f})\nValue: {value2:.3f}", 
                                   state=tk.NORMAL)
            self.current_choice_var.set("Please make a choice...")
            
            # Wait for choice
            while self.waiting_for_choice and self.is_running:
                self.root.update()
                time.sleep(0.1)
            
            return self.user_choice
        else:
            # Auto mode
            return 1 if value1 > value2 else -1
        
    def find_best_query(self, gp):
        """Find optimal query using the GP model."""
        def negative_info_gain(x):
            return -1 * gp.objectiveEntropy(x)
            
        x0 = np.array(list(gp.initialPoint) * 2) + np.random.uniform(-6, 4, gp.dim * 2)
        opt_res = opt.fmin_l_bfgs_b(
            negative_info_gain, 
            x0=x0, 
            bounds=[(-5, 5)] * gp.dim * 2, 
            approx_grad=True, 
            factr=0.1, 
            iprint=-1
        )
        return opt_res[0], -opt_res[1]
        
    def simulate_user_choice(self, query1, query2, value_q1, value_q2):
        """Get user choice based on the selected mode."""
        return self.wait_for_user_choice(query1, query2, value_q1, value_q2)
    
    def generate_query_points(self):
        """Generate optimal query points for user selection."""
        if not self.current_gp:
            messagebox.showerror("Error", "Please start simulation first!")
            return
            
        # Find optimal query points
        optimal_query, info_gain = self.find_best_query(self.current_gp)
        query1 = [round(optimal_query[0], 1), round(optimal_query[1], 1)]
        query2 = [round(optimal_query[2], 1), round(optimal_query[3], 1)]
        
        # Store candidates
        self.current_query_candidates = (query1, query2)
        self.last_info_gain = info_gain
        
        # Calculate values for display
        point = np.array([query1, query2])
        value_q1 = value_q2 = 0
        
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            rv = multivariate_normal(mean, cov)
            values = weight * rv.pdf(point)
            value_q1 += values[0]
            value_q2 += values[1]
        
        # Update display
        self.query_candidates_var.set(
            f"Point 1: ({query1[0]}, {query1[1]}) [Value: {value_q1:.3f}] | "
            f"Point 2: ({query2[0]}, {query2[1]}) [Value: {value_q2:.3f}]"
        )
        
        # Update button labels and enable selection
        self.select_point1_btn.config(
            text=f"Select Point 1\n({query1[0]}, {query1[1]})\nValue: {value_q1:.3f}",
            state=tk.NORMAL
        )
        self.select_point2_btn.config(
            text=f"Select Point 2\n({query2[0]}, {query2[1]})\nValue: {value_q2:.3f}",
            state=tk.NORMAL
        )
        
        # Disable generation until next step
        self.generate_query_btn.config(state=tk.DISABLED)
        
    def select_query_point(self, point_number):
        """Select one of the generated query points."""
        if not self.current_query_candidates:
            messagebox.showerror("Error", "No query points generated!")
            return
            
        self.selected_point = point_number
        
        # Highlight selected button
        if point_number == 1:
            self.select_point1_btn.config(state=tk.DISABLED, text="✓ SELECTED\n" + self.select_point1_btn.cget("text").split('\n', 1)[1])
            self.select_point2_btn.config(state=tk.NORMAL)
        else:
            self.select_point2_btn.config(state=tk.DISABLED, text="✓ SELECTED\n" + self.select_point2_btn.cget("text").split('\n', 1)[1])
            self.select_point1_btn.config(state=tk.NORMAL)
        
        # Enable next step
        self.next_step_btn.config(state=tk.NORMAL)
        
    def next_interactive_step(self):
        """Execute the next interactive step with selected point."""
        if not self.selected_point or not self.current_query_candidates:
            messagebox.showerror("Error", "Please select a query point first!")
            return
            
        query1, query2 = self.current_query_candidates
        uncertainty_level = self.manual_uncertainty_var.get()
        
        # Calculate values for both points
        point = np.array([query1, query2]) 
        value_q1 = value_q2 = 0
        
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            rv = multivariate_normal(mean, cov)
            values = weight * rv.pdf(point)
            value_q1 += values[0]
            value_q2 += values[1]
        
        # The selected point becomes the "preferred" one
        # Point 1 selected means query1 > query2, so user_choice = 1
        # Point 2 selected means query2 > query1, so user_choice = -1
        user_choice = 1 if self.selected_point == 1 else -1
        
        # Update GUI with current query and uncertainty
        diff = np.linalg.norm(value_q1 - value_q2)
        self.update_status(query1, query2, uncertainty_level, diff)
        
        # Update preference dictionary
        a = (query1[0], query1[1])
        b = (query2[0], query2[1])
        self.current_pref_dict[a] = self.current_pref_dict.get(a, 0) + self.uncertainty_dict[uncertainty_level]
        self.current_pref_dict[b] = self.current_pref_dict.get(b, 0) + self.uncertainty_dict[uncertainty_level]
        
        # Update GP
        self.current_gp.updateParameters(
            [query1, query2],
            user_choice,
            uncertainty_level,
            self.current_pref_dict
        )
        
        # Store data - store the selected point
        selected_query = query1 if self.selected_point == 1 else query2
        self.query_history.append(selected_query)
        self.uncertainty_history.append(uncertainty_level)
        self.current_iteration += 1
        
        # Calculate correlation
        corr = self.get_correlation(self.current_gp)
        self.correlation_history.append(corr)
        self.info_gain_history.append(self.last_info_gain if hasattr(self, 'last_info_gain') else 0)
        
        # Update plots
        self.update_plots(self.current_gp)
        
        # Update progress
        progress = (self.current_iteration / self.max_iterations) * 100
        self.progress_var.set(progress)
        
        # Reset for next step
        self.current_query_candidates = None
        self.selected_point = None
        self.query_candidates_var.set("Not generated yet")
        self.select_point1_btn.config(state=tk.DISABLED, text="Select Point 1")
        self.select_point2_btn.config(state=tk.DISABLED, text="Select Point 2")
        self.next_step_btn.config(state=tk.DISABLED)
        self.generate_query_btn.config(state=tk.NORMAL)
            
    def get_correlation(self, gp):
        """Calculate correlation between GP predictions and true GMM."""
        pos = np.dstack((self.x_grid, self.y_grid))
        x_pred = pos.reshape(-1, 2)
        y_pred = gp.mean1pt(x_pred, eval=True)
        corr = np.corrcoef(self.gmm_pdf.flatten(), y_pred)[0, 1]
        return corr
        
    def simulation_step(self, gp, pref_dict):
        """Perform one simulation step."""
        # Find optimal query
        optimal_query, info_gain = self.find_best_query(gp)
        next_query_1 = [float(round(optimal_query[0], 1)), float(round(optimal_query[1], 1))]
        next_query_2 = [float(round(optimal_query[2], 1)), float(round(optimal_query[3], 1))]
        
        # Calculate values for both points
        point = np.array([next_query_1, next_query_2])
        value_q1 = value_q2 = 0
        
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            rv = multivariate_normal(mean, cov)
            values = weight * rv.pdf(point)
            value_q1 += values[0]
            value_q2 += values[1]
        
        # Determine uncertainty level
        diff = np.linalg.norm(value_q1 - value_q2)
        uncertainty_level = 1
        for j, thresh in enumerate(self.uncertainty_thresh, 1):
            if diff > thresh:
                uncertainty_level = j + 1
        
        # Update GUI with current query and uncertainty
        self.root.after(0, lambda: self.update_status(next_query_1, next_query_2, uncertainty_level, diff))
        
        # Get user choice
        user_choice = self.simulate_user_choice(next_query_1, next_query_2, value_q1, value_q2)
        
        # Update preference dictionary
        a = (next_query_1[0], next_query_1[1])
        b = (next_query_2[0], next_query_2[1])
        pref_dict[a] = pref_dict.get(a, 0) + self.uncertainty_dict[uncertainty_level]
        pref_dict[b] = pref_dict.get(b, 0) + self.uncertainty_dict[uncertainty_level]
        
        # Update GP
        gp.updateParameters(
            [next_query_1, next_query_2],
            user_choice,
            uncertainty_level,
            pref_dict
        )
        
        # Store data
        self.query_history.append(next_query_1)
        self.info_gain_history.append(info_gain)
        self.uncertainty_history.append(uncertainty_level)
        
        # Calculate correlation
        corr = self.get_correlation(gp)
        self.correlation_history.append(corr)
        
        return gp, pref_dict
        
    def update_status(self, query1, query2, uncertainty_level, diff):
        """Update status display."""
        # Update uncertainty display
        uncertainty_colors = {1: "green", 2: "yellow", 3: "orange", 4: "red", 5: "darkred"}
        self.uncertainty_var.set(f"Level {uncertainty_level} (diff: {diff:.3f})")
        self.uncertainty_label.config(foreground=uncertainty_colors.get(uncertainty_level, "black"))
        
        # Update query display
        self.query_var.set(f"Q1: ({query1[0]:.1f}, {query1[1]:.1f}), Q2: ({query2[0]:.1f}, {query2[1]:.1f})")
        
    def run_simulation(self):
        """Main simulation loop."""
        # Initialize
        np.random.seed(47)
        gp = GaussianProcess(self.initial_point, self.theta, self.noise_level)
        pref_dict = {(2, 2): 1, (0, 0): 1}
        gp.updateParameters([[0, 0], [2, 2]], -1, 5, pref_dict)
        
        self.current_iteration = 0
        
        while self.is_running and self.current_iteration < self.max_iterations:
            # Perform simulation step
            gp, pref_dict = self.simulation_step(gp, pref_dict)
            self.current_iteration += 1
            
            # Update GUI
            self.root.after(0, lambda: self.update_plots(gp))
            
            # Update progress
            progress = (self.current_iteration / self.max_iterations) * 100
            self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            # Control simulation speed
            time.sleep(1.0 / self.speed_var.get())
            
        # Simulation completed
        self.root.after(0, self.simulation_finished)
        
    def update_plots(self, gp):
        """Update all plots."""
        self.update_surface_plot(gp)
        self.update_convergence_plot()
        self.update_query_plot()
        
    def start_simulation(self):
        """Start the simulation."""
        if not self.is_running:
            # Initialize GP and preference dictionary
            np.random.seed(47)
            self.current_gp = GaussianProcess(self.initial_point, self.theta, self.noise_level)
            self.current_pref_dict = {(2, 2): 1, (0, 0): 1}
            self.current_gp.updateParameters([[0, 0], [2, 2]], -1, 5, self.current_pref_dict)
            
            # Check simulation mode
            if self.method_var.get() == "Interactive":
                # Enable interactive controls
                self.generate_query_btn.config(state=tk.NORMAL)
                self.manual_mode = True
                self.start_btn.config(text="Initialize", state=tk.DISABLED)
                messagebox.showinfo("Interactive Mode", 
                                   "Interactive mode enabled.\n\n" +
                                   "1. Click 'Generate Query Points' to get optimal candidates\n" +
                                   "2. Select one of the two points\n" +
                                   "3. Set uncertainty level and click 'Next Step'")
            else:
                # Start automatic simulation
                self.is_running = True
                self.start_btn.config(state=tk.DISABLED)
                self.pause_btn.config(state=tk.NORMAL)
                
                # Start simulation in separate thread
                self.simulation_thread = threading.Thread(target=self.run_simulation)
                self.simulation_thread.daemon = True
                self.simulation_thread.start()
            
    def pause_simulation(self):
        """Pause the simulation."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
    def reset_simulation(self):
        """Reset the simulation."""
        self.is_running = False
        self.manual_mode = False
        self.current_iteration = 0
        self.correlation_history = []
        self.query_history = []
        self.info_gain_history = []
        self.uncertainty_history = []
        self.progress_var.set(0)
        
        # Reset GP and preference dictionary
        self.current_gp = None
        self.current_pref_dict = {}
        
        # Reset choice buttons
        self.waiting_for_choice = False
        self.user_choice = None
        self.choice1_btn.config(state=tk.DISABLED, text="Choice 1")
        self.choice2_btn.config(state=tk.DISABLED, text="Choice 2")
        self.current_choice_var.set("Waiting...")
        
        # Reset interactive controls
        self.generate_query_btn.config(state=tk.DISABLED)
        self.select_point1_btn.config(state=tk.DISABLED, text="Select Point 1")
        self.select_point2_btn.config(state=tk.DISABLED, text="Select Point 2")
        self.next_step_btn.config(state=tk.DISABLED)
        self.query_candidates_var.set("Not generated yet")
        
        # Reset interactive state
        self.current_query_candidates = None
        self.selected_point = None
        self.waiting_for_point_selection = False
        
        # Reset status display
        self.uncertainty_var.set("N/A")
        self.query_var.set("N/A")
        
        self.start_btn.config(state=tk.NORMAL, text="Start Simulation")
        self.pause_btn.config(state=tk.DISABLED)
        
        # Reset plots
        self.create_gmm_surface()
        self.update_convergence_plot()
        self.update_query_plot()
        
    def simulation_finished(self):
        """Handle simulation completion."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Simulation Complete", 
                           f"Simulation completed after {self.current_iteration} iterations!")


def main():
    """Main function to run the interactive interface."""
    print("UUPL Interactive Interface")
    print("=" * 40)
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print("Checking dependencies...")
        if not install_missing_packages(missing):
            print("Error: Could not install all required dependencies.")
            print("Please install the following packages manually:")
            for pkg in missing:
                print(f"  - {pkg}")
            return 1
    
    # Launch the interface
    try:
        print("Launching UUPL Interactive Interface...")
        root = tk.Tk()
        UUPLInteractiveInterface(root)
        root.mainloop()
    except ImportError as e:
        print(f"Error importing interface: {e}")
        print("Make sure all required files are in the same directory.")
        return 1
    except Exception as e:
        print(f"Error launching interface: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())