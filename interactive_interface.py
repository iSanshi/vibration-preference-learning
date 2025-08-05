#!/usr/bin/env python3
"""
UUPL Interactive Interface - Redesigned

Simplified and cleaner interface for UUPL simulation with only essential controls.
"""

import sys
import subprocess
import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox
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
    required_packages = ['tkinter', 'matplotlib', 'numpy', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        if package == 'tkinter':
            try:
                import tkinter as _
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
        for package in packages:
            if package == 'tkinter':
                print("tkinter is built-in. Please ensure complete Python installation.")
                continue
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install manually.")
                return False
    return True


class UUPLInterface:
    """Simplified UUPL Interactive Interface."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("UUPL Interactive Interface")
        self.root.geometry("1200x800")
        
        # Initialize parameters
        self.setup_parameters()
        self.setup_variables()
        
        # Create GUI
        self.create_widgets()
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
        self.uncertainty_dict = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
        self.uncertainty_thresh = [0.01, 0.05, 0.1, 0.2]
        
    def setup_variables(self):
        """Initialize control variables."""
        # Simulation state
        self.current_iteration = 0
        self.max_iterations = 50
        self.is_running = False
        self.is_interactive = True
        
        # Data storage
        self.correlation_history = []
        self.query_history = []
        self.info_gain_history = []
        self.uncertainty_history = []
        
        # GP and preferences
        self.current_gp = None
        self.current_pref_dict = {}
        
        # Interactive control
        self.current_candidates = None
        self.selected_point = None
        
    def create_widgets(self):
        """Create GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Plot area
        self.create_plot_area(main_frame)
        
    def create_control_panel(self, parent):
        """Create the control panel."""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="Interactive")
        ttk.Radiobutton(mode_frame, text="Interactive", variable=self.mode_var, 
                       value="Interactive", command=self.on_mode_change).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Radiobutton(mode_frame, text="Automatic", variable=self.mode_var, 
                       value="Automatic", command=self.on_mode_change).pack(side=tk.LEFT)
        
        # Interactive controls
        self.interactive_frame = ttk.LabelFrame(control_frame, text="Interactive Controls", padding=5)
        self.interactive_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Candidate display
        self.candidates_var = tk.StringVar(value="Click 'Generate' to start")
        ttk.Label(self.interactive_frame, textvariable=self.candidates_var, 
                 wraplength=400).pack(pady=(0, 5))
        
        # Selection buttons
        selection_frame = ttk.Frame(self.interactive_frame)
        selection_frame.pack(pady=(0, 5))
        
        self.select1_btn = ttk.Button(selection_frame, text="Select Point 1", 
                                     command=lambda: self.select_point(1), state=tk.DISABLED)
        self.select1_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.select2_btn = ttk.Button(selection_frame, text="Select Point 2", 
                                     command=lambda: self.select_point(2), state=tk.DISABLED)
        self.select2_btn.pack(side=tk.LEFT)
        
        # Uncertainty and controls
        control_row = ttk.Frame(self.interactive_frame)
        control_row.pack(fill=tk.X)
        
        ttk.Label(control_row, text="Uncertainty:").pack(side=tk.LEFT)
        self.uncertainty_var = tk.IntVar(value=1)
        ttk.Scale(control_row, from_=1, to=5, orient=tk.HORIZONTAL, 
                 variable=self.uncertainty_var, length=100).pack(side=tk.LEFT, padx=(5, 20))
        
        self.generate_btn = ttk.Button(control_row, text="Generate", 
                                      command=self.generate_candidates, state=tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.execute_btn = ttk.Button(control_row, text="Execute", 
                                     command=self.execute_step, state=tk.DISABLED)
        self.execute_btn.pack(side=tk.LEFT)
        
        # Main controls
        main_controls = ttk.Frame(control_frame)
        main_controls.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(main_controls, text="Start", command=self.start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_btn = ttk.Button(main_controls, text="Pause", command=self.pause_simulation, 
                                   state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_btn = ttk.Button(main_controls, text="Reset", command=self.reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Status display
        status_frame = ttk.Frame(main_controls)
        status_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(status_frame, text="Iteration:").pack(side=tk.LEFT)
        self.iteration_var = tk.StringVar(value="0")
        ttk.Label(status_frame, textvariable=self.iteration_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(status_frame, text="Correlation:").pack(side=tk.LEFT)
        self.correlation_var = tk.StringVar(value="N/A")
        ttk.Label(status_frame, textvariable=self.correlation_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(5, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_controls, variable=self.progress_var, 
                                          maximum=100, length=150)
        self.progress_bar.pack(side=tk.RIGHT)
        
    def create_plot_area(self, parent):
        """Create the plotting area."""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Surface tab
        self.surface_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.surface_frame, text="Function Surface")
        
        # Convergence tab
        self.convergence_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.convergence_frame, text="Convergence")
        
    def setup_plots(self):
        """Setup matplotlib plots."""
        # Surface plot
        self.surface_fig = Figure(figsize=(12, 6))
        self.surface_canvas = FigureCanvasTkAgg(self.surface_fig, self.surface_frame)
        self.surface_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Convergence plot
        self.convergence_fig = Figure(figsize=(12, 6))
        self.convergence_canvas = FigureCanvasTkAgg(self.convergence_fig, self.convergence_frame)
        self.convergence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.create_gmm_surface()
        self.update_convergence_plot()
        
    def on_mode_change(self):
        """Handle mode change."""
        self.is_interactive = (self.mode_var.get() == "Interactive")
        if self.is_interactive:
            self.interactive_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.interactive_frame.pack_forget()
            
    def create_gmm_surface(self):
        """Create the GMM surface visualization."""
        step_size = 0.1
        x, y = np.mgrid[-5:5+step_size:step_size, -5:5+step_size:step_size]
        pos = np.dstack((x, y))
        
        # Calculate GMM PDF
        gmm_pdf = np.zeros(x.shape)
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            rv = multivariate_normal(mean, cov)
            gmm_pdf += weight * rv.pdf(pos)
        
        self.x_grid, self.y_grid, self.gmm_pdf = x, y, gmm_pdf
        
        # Plot
        self.surface_fig.clear()
        ax1 = self.surface_fig.add_subplot(121, projection='3d')
        ax1.plot_surface(x, y, gmm_pdf, cmap='viridis', alpha=0.8)
        ax1.set_title('True Function (GMM)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Value')
        ax1.view_init(elev=30, azim=45)
        
        # Placeholder for learned surface
        ax2 = self.surface_fig.add_subplot(122, projection='3d')
        ax2.set_title('Learned Function')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Value')
        ax2.view_init(elev=30, azim=45)
        
        self.surface_fig.tight_layout()
        self.surface_canvas.draw()
        
    def update_surface_plot(self):
        """Update the learned surface plot."""
        if not self.current_gp:
            return
            
        pos = np.dstack((self.x_grid, self.y_grid))
        x_pred = pos.reshape(-1, 2)
        y_pred = self.current_gp.mean1pt(x_pred, eval=True)
        y_pred = y_pred.reshape(self.x_grid.shape)
        
        # Update the second subplot
        ax2 = self.surface_fig.axes[1]
        ax2.clear()
        ax2.plot_surface(self.x_grid, self.y_grid, y_pred, cmap='plasma', alpha=0.8)
        ax2.set_title(f'Learned Function (Iter {self.current_iteration})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Value')
        ax2.view_init(elev=30, azim=45)
        
        # Add query points
        if self.query_history:
            queries = np.array(self.query_history)
            pred_values = [self.current_gp.mean1pt(q.reshape(1, -1), eval=True) for q in queries]
            ax2.scatter(queries[:, 0], queries[:, 1], pred_values, c='red', s=50)
        
        self.surface_canvas.draw()
        
    def update_convergence_plot(self):
        """Update convergence plots."""
        self.convergence_fig.clear()
        
        # Correlation plot
        ax1 = self.convergence_fig.add_subplot(221)
        if self.correlation_history:
            ax1.plot(self.correlation_history, 'b-', linewidth=2)
            ax1.set_title('Correlation with True Function')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Correlation')
            ax1.grid(True, alpha=0.3)
        
        # Information gain plot
        ax2 = self.convergence_fig.add_subplot(222)
        if self.info_gain_history:
            ax2.plot(self.info_gain_history, 'g-', linewidth=2)
            ax2.set_title('Information Gain')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Info Gain')
            ax2.grid(True, alpha=0.3)
        
        # Query distribution
        ax3 = self.convergence_fig.add_subplot(223)
        if self.query_history:
            queries = np.array(self.query_history)
            ax3.scatter(queries[:, 0], queries[:, 1], c=range(len(queries)), 
                       cmap='viridis', s=50, alpha=0.7)
            ax3.set_title('Query Points')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.grid(True, alpha=0.3)
        
        # Uncertainty levels
        ax4 = self.convergence_fig.add_subplot(224)
        if self.uncertainty_history:
            ax4.plot(self.uncertainty_history, 'r-o', linewidth=2, markersize=4)
            ax4.set_title('Uncertainty Levels')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Uncertainty Level')
            ax4.set_ylim(0.5, 5.5)
            ax4.grid(True, alpha=0.3)
        
        self.convergence_fig.tight_layout()
        self.convergence_canvas.draw()
        
    def find_best_query(self):
        """Find optimal query points."""
        def negative_info_gain(x):
            return -1 * self.current_gp.objectiveEntropy(x)
            
        x0 = np.array(list(self.current_gp.initialPoint) * 2) + np.random.uniform(-6, 4, self.current_gp.dim * 2)
        opt_res = opt.fmin_l_bfgs_b(
            negative_info_gain, 
            x0=x0, 
            bounds=[(-5, 5)] * self.current_gp.dim * 2, 
            approx_grad=True, 
            factr=0.1, 
            iprint=-1
        )
        return opt_res[0], -opt_res[1]
        
    def calculate_point_values(self, point1, point2):
        """Calculate GMM values for two points."""
        points = np.array([point1, point2])
        values = np.zeros(2)
        
        for mean, cov, weight in zip(self.means, self.covariances, self.weights):
            rv = multivariate_normal(mean, cov)
            values += weight * rv.pdf(points)
            
        return values
        
    def get_correlation(self):
        """Calculate correlation with true function."""
        if not self.current_gp:
            return 0
            
        pos = np.dstack((self.x_grid, self.y_grid))
        x_pred = pos.reshape(-1, 2)
        y_pred = self.current_gp.mean1pt(x_pred, eval=True)
        return np.corrcoef(self.gmm_pdf.flatten(), y_pred)[0, 1]
        
    def generate_candidates(self):
        """Generate candidate query points."""
        if not self.current_gp:
            messagebox.showerror("Error", "Please start simulation first!")
            return
            
        # Find optimal query points
        optimal_query, info_gain = self.find_best_query()
        point1 = [round(optimal_query[0], 1), round(optimal_query[1], 1)]
        point2 = [round(optimal_query[2], 1), round(optimal_query[3], 1)]
        
        # Calculate values
        values = self.calculate_point_values(point1, point2)
        
        # Store candidates
        self.current_candidates = (point1, point2, values, info_gain)
        
        # Update display
        self.candidates_var.set(
            f"Point 1: ({point1[0]}, {point1[1]}) [Value: {values[0]:.3f}] | "
            f"Point 2: ({point2[0]}, {point2[1]}) [Value: {values[1]:.3f}]"
        )
        
        # Enable selection
        self.select1_btn.config(state=tk.NORMAL)
        self.select2_btn.config(state=tk.NORMAL)
        self.generate_btn.config(state=tk.DISABLED)
        
    def select_point(self, point_num):
        """Select a query point."""
        self.selected_point = point_num
        
        # Update button states
        if point_num == 1:
            self.select1_btn.config(text="✓ Point 1 Selected", state=tk.DISABLED)
            self.select2_btn.config(text="Select Point 2", state=tk.NORMAL)
        else:
            self.select2_btn.config(text="✓ Point 2 Selected", state=tk.DISABLED)
            self.select1_btn.config(text="Select Point 1", state=tk.NORMAL)
            
        # Enable execute
        self.execute_btn.config(state=tk.NORMAL)
        
    def execute_step(self):
        """Execute the selected step."""
        if not self.current_candidates or not self.selected_point:
            messagebox.showerror("Error", "Please select a point first!")
            return
            
        point1, point2, values, info_gain = self.current_candidates
        uncertainty_level = self.uncertainty_var.get()
        
        # User choice based on selection
        user_choice = 1 if self.selected_point == 1 else -1
        
        # Update preference dictionary
        a = (point1[0], point1[1])
        b = (point2[0], point2[1])
        self.current_pref_dict[a] = self.current_pref_dict.get(a, 0) + self.uncertainty_dict[uncertainty_level]
        self.current_pref_dict[b] = self.current_pref_dict.get(b, 0) + self.uncertainty_dict[uncertainty_level]
        
        # Update GP
        self.current_gp.updateParameters(
            [point1, point2],
            user_choice,
            uncertainty_level,
            self.current_pref_dict
        )
        
        # Store data
        selected_query = point1 if self.selected_point == 1 else point2
        self.query_history.append(selected_query)
        self.uncertainty_history.append(uncertainty_level)
        self.info_gain_history.append(info_gain)
        self.current_iteration += 1
        
        # Calculate correlation
        corr = self.get_correlation()
        self.correlation_history.append(corr)
        
        # Update displays
        self.update_status()
        self.update_plots()
        
        # Reset for next step
        self.reset_interactive_controls()
        
    def update_status(self):
        """Update status displays."""
        self.iteration_var.set(str(self.current_iteration))
        if self.correlation_history:
            self.correlation_var.set(f"{self.correlation_history[-1]:.3f}")
        progress = (self.current_iteration / self.max_iterations) * 100
        self.progress_var.set(progress)
        
    def update_plots(self):
        """Update all plots."""
        self.update_surface_plot()
        self.update_convergence_plot()
        
    def reset_interactive_controls(self):
        """Reset interactive controls for next step."""
        self.current_candidates = None
        self.selected_point = None
        self.candidates_var.set("Click 'Generate' to create next query")
        self.select1_btn.config(text="Select Point 1", state=tk.DISABLED)
        self.select2_btn.config(text="Select Point 2", state=tk.DISABLED)
        self.execute_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.NORMAL)
        
    def start_simulation(self):
        """Start the simulation."""
        # Initialize GP
        np.random.seed(47)
        self.current_gp = GaussianProcess(self.initial_point, self.theta, self.noise_level)
        self.current_pref_dict = {(2, 2): 1, (0, 0): 1}
        self.current_gp.updateParameters([[0, 0], [2, 2]], -1, 5, self.current_pref_dict)
        
        if self.is_interactive:
            # Interactive mode
            self.generate_btn.config(state=tk.NORMAL)
            self.start_btn.config(text="Initialize", state=tk.DISABLED)
            messagebox.showinfo("Interactive Mode", "Click 'Generate' to start!")
        else:
            # Automatic mode
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            
            # Start simulation in thread
            self.simulation_thread = threading.Thread(target=self.run_automatic_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
    def run_automatic_simulation(self):
        """Run automatic simulation."""
        while self.is_running and self.current_iteration < self.max_iterations:
            # Generate and execute automatically
            optimal_query, info_gain = self.find_best_query()
            point1 = [round(optimal_query[0], 1), round(optimal_query[1], 1)]
            point2 = [round(optimal_query[2], 1), round(optimal_query[3], 1)]
            
            values = self.calculate_point_values(point1, point2)
            
            # Auto select best point
            user_choice = 1 if values[0] > values[1] else -1
            selected_query = point1 if user_choice == 1 else point2
            
            # Auto determine uncertainty
            diff = np.abs(values[0] - values[1])
            uncertainty_level = 1
            for j, thresh in enumerate(self.uncertainty_thresh, 1):
                if diff > thresh:
                    uncertainty_level = j + 1
                    
            # Update model
            a = (point1[0], point1[1])
            b = (point2[0], point2[1])
            self.current_pref_dict[a] = self.current_pref_dict.get(a, 0) + self.uncertainty_dict[uncertainty_level]
            self.current_pref_dict[b] = self.current_pref_dict.get(b, 0) + self.uncertainty_dict[uncertainty_level]
            
            self.current_gp.updateParameters([point1, point2], user_choice, uncertainty_level, self.current_pref_dict)
            
            # Store data
            self.query_history.append(selected_query)
            self.uncertainty_history.append(uncertainty_level)
            self.info_gain_history.append(info_gain)
            self.current_iteration += 1
            
            corr = self.get_correlation()
            self.correlation_history.append(corr)
            
            # Update GUI
            self.root.after(0, self.update_status)
            self.root.after(0, self.update_plots)
            
            time.sleep(0.5)  # Control speed
            
        # Simulation finished
        self.root.after(0, self.simulation_finished)
        
    def pause_simulation(self):
        """Pause automatic simulation."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
    def reset_simulation(self):
        """Reset the simulation."""
        self.is_running = False
        self.current_iteration = 0
        self.correlation_history = []
        self.query_history = []
        self.info_gain_history = []
        self.uncertainty_history = []
        
        self.current_gp = None
        self.current_pref_dict = {}
        
        self.reset_interactive_controls()
        
        self.start_btn.config(text="Start", state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.DISABLED)
        
        self.update_status()
        self.create_gmm_surface()
        self.update_convergence_plot()
        
    def simulation_finished(self):
        """Handle simulation completion."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Complete", f"Simulation finished after {self.current_iteration} iterations!")


def main():
    """Main function."""
    print("UUPL Interactive Interface")
    print("=" * 40)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("Checking dependencies...")
        if not install_missing_packages(missing):
            print("Error: Could not install all dependencies.")
            return 1
    
    # Launch interface
    try:
        print("Launching interface...")
        root = tk.Tk()
        UUPLInterface(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())