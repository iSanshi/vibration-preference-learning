"""
Four-dimensional extension of GP_ours for audio signal optimization.
Optimizes 4 parameters: amplitude, frequency, density, gradient.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
import sys
import os

# Add parent directory to path to import GP_ours
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GP_ours import GaussianProcess
from util import *


class GaussianProcess4D(GaussianProcess):
    """4D extension of GaussianProcess for audio parameter optimization."""
    
    def __init__(self, initial_point=None, theta=0.5, noise_level=0.1):
        """
        Initialize 4D Gaussian Process.
        
        Parameters:
        -----------
        initial_point : list or array
            Initial 4D point [amplitude, frequency, density, gradient] (normalized to [0,1])
        theta : float
            Length scale parameter for kernel
        noise_level : float
            Noise level parameter
        """
        if initial_point is None:
            initial_point = [0.5, 0.5, 0.5, 0.5]  # Center of normalized space
            
        super().__init__(initial_point, theta, noise_level)
        self.dim = 4  # Override dimension
        self.parameter_bounds = {
            'amplitude': [30, 60],
            'frequency': [25, 75], 
            'density': [10, 90],
            'gradient': [-50, 50]
        }
        
    def normalize_parameters(self, params):
        """
        Normalize parameters from physical ranges to [0,1].
        
        Parameters:
        -----------
        params : array-like
            [amplitude, frequency, density, gradient] in physical units
            
        Returns:
        --------
        normalized : ndarray
            Normalized parameters in [0,1]
        """
        params = np.array(params)
        normalized = np.zeros(4)
        
        # Amplitude: [30, 60] -> [0, 1]
        normalized[0] = (params[0] - 30) / (60 - 30)
        
        # Frequency: [25, 75] -> [0, 1]
        normalized[1] = (params[1] - 25) / (75 - 25)
        
        # Density: [10, 90] -> [0, 1]
        normalized[2] = (params[2] - 10) / (90 - 10)
        
        # Gradient: [-50, 50] -> [0, 1]
        normalized[3] = (params[3] - (-50)) / (50 - (-50))
        
        return normalized
    
    def denormalize_parameters(self, normalized_params):
        """
        Convert normalized parameters back to physical ranges.
        
        Parameters:
        -----------
        normalized_params : array-like
            Parameters in [0,1] range
            
        Returns:
        --------
        physical : ndarray
            Parameters in physical units
        """
        normalized_params = np.array(normalized_params)
        physical = np.zeros(4)
        
        # Amplitude: [0, 1] -> [30, 60]
        physical[0] = normalized_params[0] * (60 - 30) + 30
        
        # Frequency: [0, 1] -> [25, 75]
        physical[1] = normalized_params[1] * (75 - 25) + 25
        
        # Density: [0, 1] -> [10, 90]
        physical[2] = normalized_params[2] * (90 - 10) + 10
        
        # Gradient: [0, 1] -> [-50, 50]
        physical[3] = normalized_params[3] * (50 - (-50)) + (-50)
        
        return physical
    
    def objectiveEntropy4D(self, x):
        """
        4D version of objective entropy for a query [xa, xb] where each point is 4D.
        
        Parameters:
        -----------
        x : ndarray
            Concatenated array of two 4D points [xa, xb] (length 8)
            
        Returns:
        --------
        entropy : float
            Entropy value for the query pair
        """
        xa = x[:4]  # First 4D point
        xb = x[4:]  # Second 4D point
        
        # Ensure points are in [0,1] bounds
        xa = np.clip(xa, 0, 1)
        xb = np.clip(xb, 0, 1)
        
        matCov = self.postcov(xa, xb)
        mua, mub = self.postmean(xa, xb)
        sigmap = np.sqrt(np.pi * np.log(2) / 2) * self.noise

        result1 = h(phi((mua - mub) / (np.sqrt(2*self.noise**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))))
        result2 = sigmap * 1 / (np.sqrt(sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1])) * np.exp(
            -0.5 * (mua - mub)**2 / (sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))

        return result1 - result2
    
    def find_optimal_query_4d(self):
        """
        Find optimal query points in 4D space.
        
        Returns:
        --------
        optimal_points : ndarray
            Optimal query points [xa, xb] (length 8)
        info_gain : float
            Information gain from this query
        """
        def negative_info_gain(x):
            return -1 * self.objectiveEntropy4D(x)
        
        # Random initialization in normalized space [0,1]^8
        x0 = np.random.uniform(0, 1, 8)
        
        # Bounds for normalized space
        bounds = [(0, 1)] * 8
        
        # Optimize
        opt_res = minimize(
            negative_info_gain,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return opt_res.x, -opt_res.fun
    
    def kernel_4d(self, xa, xb):
        """
        4D RBF kernel function.
        
        Parameters:
        -----------
        xa, xb : array-like
            4D points
            
        Returns:
        --------
        kernel_value : float
        """
        xa = np.array(xa)
        xb = np.array(xb)
        
        # RBF kernel with theta as length scale
        distance = np.linalg.norm(xa - xb)
        return np.exp(-self.theta * distance ** 2)
    
    def kernel(self, xa, xb):
        """Override parent kernel method to use 4D version."""
        return self.kernel_4d(xa, xb)
    
    def batch_kernel_4d(self, xa, xb):
        """
        Batch kernel computation for 4D points.
        
        Parameters:
        -----------
        xa : ndarray
            Array of 4D points (N x 4)
        xb : ndarray
            Single 4D point
            
        Returns:
        --------
        kernel_values : ndarray
            Array of kernel values
        """
        xa = np.array(xa)
        xb = np.array(xb)
        
        if xa.ndim == 1:
            xa = xa.reshape(1, -1)
        
        # Compute distances
        distances = np.linalg.norm(xa - xb, axis=1)
        return np.exp(-self.theta * distances ** 2)
    
    def batch_kernel(self, xa, xb):
        """Override parent batch_kernel method."""
        return self.batch_kernel_4d(xa, xb)


def test_gp_4d():
    """Test the 4D Gaussian Process implementation."""
    print("Testing 4D Gaussian Process...")
    
    # Initialize
    gp = GaussianProcess4D()
    
    # Test parameter normalization
    physical_params = [45, 50, 50, 0]  # Middle values
    normalized = gp.normalize_parameters(physical_params)
    denormalized = gp.denormalize_parameters(normalized)
    
    print(f"Original: {physical_params}")
    print(f"Normalized: {normalized}")
    print(f"Denormalized: {denormalized}")
    
    # Test with some dummy preferences
    pref_dict = {tuple(normalized): 1.0}
    gp.updateParameters([normalized, normalized], 0, 1, pref_dict)
    
    # Test optimal query finding
    try:
        optimal_query, info_gain = gp.find_optimal_query_4d()
        print(f"Optimal query found with info gain: {info_gain}")
        print(f"Query points shape: {optimal_query.shape}")
        
        # Convert back to physical parameters
        point1_phys = gp.denormalize_parameters(optimal_query[:4])
        point2_phys = gp.denormalize_parameters(optimal_query[4:])
        
        print(f"Point 1 (physical): {point1_phys}")
        print(f"Point 2 (physical): {point2_phys}")
        
    except Exception as e:
        print(f"Error in optimal query: {e}")
    
    print("4D GP test completed!")


if __name__ == "__main__":
    test_gp_4d()