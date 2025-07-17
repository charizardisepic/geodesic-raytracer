"""
Geodesic Raytracer - A General Relativity Based Black Hole Visualization System

This module implements the core physics for computing geodesics (light ray paths)
in curved spacetime around black holes using the Schwarzschild metric.
"""

import numpy as np
from numba import njit, prange
import concurrent.futures
from typing import Tuple, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@njit
def schwarzschild_metric_components(r: float, theta: float, M: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Calculate the components of the Schwarzschild metric tensor.
    
    The Schwarzschild metric describes spacetime around a spherically symmetric mass:
    ds² = -(1-2M/r)dt² + (1-2M/r)⁻¹dr² + r²dθ² + r²sin²θdφ²
    
    Args:
        r: Radial coordinate (should be > 2M to avoid singularity)
        theta: Polar angle
        M: Mass of the black hole (in geometric units, default=1)
        
    Returns:
        Tuple of metric components (g_tt, g_rr, g_theta_theta, g_phi_phi)
    """
    rs = 2.0 * M  # Schwarzschild radius
    
    if r <= rs:
        # Inside event horizon - handle carefully
        r = rs + 1e-10
    
    factor = 1.0 - rs / r
    sin_theta = np.sin(theta)
    
    g_tt = -factor
    g_rr = 1.0 / factor
    g_theta_theta = r * r
    g_phi_phi = r * r * sin_theta * sin_theta
    
    return g_tt, g_rr, g_theta_theta, g_phi_phi

@njit
def christoffel_symbols(r: float, theta: float, M: float = 1.0) -> np.ndarray:
    """
    Calculate the Christoffel symbols for the Schwarzschild metric.
    
    The Christoffel symbols encode the curvature of spacetime and are essential
    for computing geodesic equations.
    
    Args:
        r: Radial coordinate
        theta: Polar angle  
        M: Mass of the black hole
        
    Returns:
        4x4x4 array of Christoffel symbols Γᵢⱼₖ
    """
    rs = 2.0 * M
    
    if r <= rs:
        r = rs + 1e-10
        
    # Initialize Christoffel symbols array
    gamma = np.zeros((4, 4, 4))
    
    # Common factors
    factor = 1.0 - rs / r
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Non-zero Christoffel symbols for Schwarzschild metric
    
    # Γᵗᵗʳ = Γᵗʳᵗ
    gamma[0, 0, 1] = gamma[0, 1, 0] = rs / (2.0 * r * r * factor)
    
    # Γʳᵗᵗ  
    gamma[1, 0, 0] = rs * factor / (2.0 * r * r)
    
    # Γʳʳʳ
    gamma[1, 1, 1] = -rs / (2.0 * r * r * factor)
    
    # Γʳᶿᶿ
    gamma[1, 2, 2] = -(r - rs)
    
    # Γʳᶠᶠ
    gamma[1, 3, 3] = -(r - rs) * sin_theta * sin_theta
    
    # Γᶿʳᶿ = Γᶿᶿʳ
    gamma[2, 1, 2] = gamma[2, 2, 1] = 1.0 / r
    
    # Γᶿᶠᶠ
    gamma[2, 3, 3] = -sin_theta * cos_theta
    
    # Γᶠʳᶠ = Γᶠᶠʳ
    gamma[3, 1, 3] = gamma[3, 3, 1] = 1.0 / r
    
    # Γᶠᶿᶠ = Γᶠᶠᶿ
    gamma[3, 2, 3] = gamma[3, 3, 2] = cos_theta / sin_theta
    
    return gamma

@njit
def geodesic_equation(tau: float, y: np.ndarray, M: float = 1.0) -> np.ndarray:
    """
    The geodesic equation for light rays in Schwarzschild spacetime.
    
    This implements the second-order differential equation:
    d²xᵘ/dτ² + Γᵘᵥₚ (dxᵥ/dτ)(dxᵖ/dτ) = 0
    
    Args:
        tau: Affine parameter (proper time)
        y: State vector [t, r, theta, phi, dt/dtau, dr/dtau, dtheta/dtau, dphi/dtau]
        M: Black hole mass
        
    Returns:
        Derivative vector dy/dtau
    """
    # Extract position and velocity
    t, r, theta, phi = y[0], y[1], y[2], y[3]
    dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau = y[4], y[5], y[6], y[7]
    
    # Avoid singularities
    rs = 2.0 * M
    if r <= rs:
        r = rs + 1e-10
    if theta <= 1e-10:
        theta = 1e-10
    elif theta >= np.pi - 1e-10:
        theta = np.pi - 1e-10
    
    # Calculate Christoffel symbols
    gamma = christoffel_symbols(r, theta, M)
    
    # Initialize acceleration components
    d2t_dtau2 = 0.0
    d2r_dtau2 = 0.0
    d2theta_dtau2 = 0.0
    d2phi_dtau2 = 0.0
    
    # Velocity vector
    velocity = np.array([dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau])
    
    # Compute acceleration using geodesic equation
    for mu in range(4):
        for nu in range(4):
            for sigma in range(4):
                if mu == 0:
                    d2t_dtau2 -= gamma[0, nu, sigma] * velocity[nu] * velocity[sigma]
                elif mu == 1:
                    d2r_dtau2 -= gamma[1, nu, sigma] * velocity[nu] * velocity[sigma]
                elif mu == 2:
                    d2theta_dtau2 -= gamma[2, nu, sigma] * velocity[nu] * velocity[sigma]
                elif mu == 3:
                    d2phi_dtau2 -= gamma[3, nu, sigma] * velocity[nu] * velocity[sigma]
    
    # Return derivative vector
    dydt = np.array([
        dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau,
        d2t_dtau2, d2r_dtau2, d2theta_dtau2, d2phi_dtau2
    ])
    
    return dydt

@njit(parallel=True)
def integrate_geodesic_numba(initial_conditions: np.ndarray, M: float, step_size: float, max_steps: int, min_r: float, max_r: float) -> Tuple[np.ndarray, bool]:
    """
    Numba-optimized geodesic integration loop (single ray).
    """
    rs = 2.0 * M
    trajectory = np.empty((max_steps, 8))
    y = initial_conditions.copy()
    tau = 0.0
    success = False
    for step in range(max_steps):
        trajectory[step, :] = y
        r = y[1]
        if r < min_r:
            return trajectory[:step+1], False
        if r > max_r:
            success = True
            return trajectory[:step+1], True
        y = runge_kutta_4_numba(y, tau, step_size, M)
        tau += step_size
        # Ensure theta stays in valid range
        if y[2] < 0:
            y[2] = -y[2]
            y[6] = -y[6]
        elif y[2] > np.pi:
            y[2] = 2*np.pi - y[2]
            y[6] = -y[6]
    return trajectory, success

@njit
def runge_kutta_4_numba(y: np.ndarray, tau: float, h: float, M: float) -> np.ndarray:
    k1 = geodesic_equation(tau, y, M)
    k2 = geodesic_equation(tau + h/2, y + h*k1/2, M)
    k3 = geodesic_equation(tau + h/2, y + h*k2/2, M)
    k4 = geodesic_equation(tau + h, y + h*k3, M)
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

class GeodesicIntegrator:
    """
    Numerical integrator for computing geodesic paths in curved spacetime.
    """
    
    def __init__(self, M: float = 1.0, step_size: float = 0.01):
        """
        Initialize the geodesic integrator.
        
        Args:
            M: Black hole mass in geometric units
            step_size: Integration step size
        """
        self.M = M
        self.step_size = step_size
        self.rs = 2.0 * M  # Schwarzschild radius
    
    def runge_kutta_4(self, y: np.ndarray, tau: float, h: float) -> np.ndarray:
        """
        Fourth-order Runge-Kutta integration step.
        
        Args:
            y: Current state vector
            tau: Current parameter value
            h: Step size
            
        Returns:
            Updated state vector
        """
        k1 = geodesic_equation(tau, y, self.M)
        k2 = geodesic_equation(tau + h/2, y + h*k1/2, self.M)
        k3 = geodesic_equation(tau + h/2, y + h*k2/2, self.M)
        k4 = geodesic_equation(tau + h, y + h*k3, self.M)
        
        return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def integrate_geodesic(self, initial_conditions: np.ndarray, max_steps: int = 10000,
                          min_r: float = None, max_r: float = 100.0) -> Tuple[np.ndarray, bool]:
        """
        Integrate a geodesic from initial conditions. Uses Numba-optimized loop.
        """
        if min_r is None:
            min_r = 1.1 * self.rs
        traj, success = integrate_geodesic_numba(initial_conditions, self.M, self.step_size, max_steps, min_r, max_r)
        return traj, success

    def batch_integrate_geodesics(self, initial_conditions_list: List[np.ndarray], max_steps: int = 10000,
                                  min_r: float = None, max_r: float = 100.0, max_workers: int = 8) -> List[Tuple[np.ndarray, bool]]:
        """
        Integrate multiple geodesics in parallel using threads.
        """
        if min_r is None:
            min_r = 1.1 * self.rs
        results = []
        def worker(ic):
            return integrate_geodesic_numba(ic, self.M, self.step_size, max_steps, min_r, max_r)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, ic) for ic in initial_conditions_list]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        return results
    
    def initial_conditions_from_observer(self, observer_pos: np.ndarray, 
                                       ray_direction: np.ndarray) -> np.ndarray:
        """
        Set up initial conditions for a light ray from observer position.
        
        Args:
            observer_pos: [t, r, theta, phi] of observer
            ray_direction: [dt, dr, dtheta, dphi] ray direction (will be normalized)
            
        Returns:
            Initial conditions array for geodesic integration
        """
        # Normalize ray direction to satisfy null geodesic condition
        # For light rays: gμν dxμ/dλ dxν/dλ = 0
        
        t, r, theta, phi = observer_pos
        dt, dr, dtheta, dphi = ray_direction
        
        # Get metric components
        g_tt, g_rr, g_theta_theta, g_phi_phi = schwarzschild_metric_components(r, theta, self.M)
        
        # Solve null condition: g_tt dt² + g_rr dr² + g_theta_theta dtheta² + g_phi_phi dphi² = 0
        # Assume dt = 1 (can always rescale affine parameter)
        dt = 1.0
        
        # The spatial part must satisfy: dr² + (r²)dtheta² + (r²sin²θ)dphi² = (1 - rs/r)
        spatial_norm_sq = dr*dr/g_rr + dtheta*dtheta/g_theta_theta + dphi*dphi/g_phi_phi
        
        if spatial_norm_sq > 0:
            # Normalize spatial components
            normalization = np.sqrt(-g_tt / spatial_norm_sq)
            dr *= normalization  
            dtheta *= normalization
            dphi *= normalization
        
        return np.array([t, r, theta, phi, dt, dr, dtheta, dphi])
