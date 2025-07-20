"""
Geodesic Raytracer - A General Relativity Based Black Hole Visualization System

This module implements the core physics for computing geodesics (light ray paths)
in curved spacetime around black holes using the Schwarzschild metric.
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings
import concurrent.futures
from numba import jit, njit

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
def geodesic_equation_accurate(y: np.ndarray, M: float = 1.0) -> np.ndarray:
    """
    Accurate geodesic equation for light rays in Schwarzschild spacetime.
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
    
    # Calculate factors
    factor = 1.0 - rs / r
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Non-zero Christoffel symbols for Schwarzschild metric (key terms)
    # Γᵗᵗʳ = Γᵗʳᵗ
    gamma_ttr = rs / (2.0 * r * r * factor)
    
    # Γʳᵗᵗ  
    gamma_rtt = rs * factor / (2.0 * r * r)
    
    # Γʳʳʳ
    gamma_rrr = -rs / (2.0 * r * r * factor)
    
    # Γʳᶿᶿ
    gamma_rtheta_theta = -(r - rs)
    
    # Γʳᶠᶠ
    gamma_rphi_phi = -(r - rs) * sin_theta * sin_theta
    
    # Γᶿʳᶿ = Γᶿᶿʳ
    gamma_theta_r_theta = 1.0 / r
    
    # Γᶿᶠᶠ
    gamma_theta_phi_phi = -sin_theta * cos_theta
    
    # Γᶠʳᶠ = Γᶠᶠʳ
    gamma_phi_r_phi = 1.0 / r
    
    # Γᶠᶿᶠ = Γᶠᶠᶿ
    gamma_phi_theta_phi = cos_theta / sin_theta
    
    # Compute accelerations
    d2t_dtau2 = -2.0 * gamma_ttr * dt_dtau * dr_dtau
    
    d2r_dtau2 = (-gamma_rtt * dt_dtau * dt_dtau - 
                 gamma_rrr * dr_dtau * dr_dtau -
                 gamma_rtheta_theta * dtheta_dtau * dtheta_dtau -
                 gamma_rphi_phi * dphi_dtau * dphi_dtau)
    
    d2theta_dtau2 = (-2.0 * gamma_theta_r_theta * dr_dtau * dtheta_dtau -
                     gamma_theta_phi_phi * dphi_dtau * dphi_dtau)
    
    d2phi_dtau2 = (-2.0 * gamma_phi_r_phi * dr_dtau * dphi_dtau -
                   2.0 * gamma_phi_theta_phi * dtheta_dtau * dphi_dtau)
    
    return np.array([
        dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau,
        d2t_dtau2, d2r_dtau2, d2theta_dtau2, d2phi_dtau2
    ])

@njit
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
        # Simplified RK4 call
        k1 = geodesic_equation_accurate(y, M)
        k2 = geodesic_equation_accurate(y + step_size*k1/2, M)
        k3 = geodesic_equation_accurate(y + step_size*k2/2, M)
        k4 = geodesic_equation_accurate(y + step_size*k3, M)
        y = y + step_size * (k1 + 2*k2 + 2*k3 + k4) / 6
        
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
    k1 = geodesic_equation_accurate(y, M)
    k2 = geodesic_equation_accurate(y + h*k1/2, M)
    k3 = geodesic_equation_accurate(y + h*k2/2, M)
    k4 = geodesic_equation_accurate(y + h*k3, M)
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

class GeodesicIntegrator:
    """
    Numerical integrator for computing geodesic paths in curved spacetime.
    """
    
    def __init__(self, M: float = 1.0, step_size: float = 0.05):
        """
        Initialize the geodesic integrator.
        
        Args:
            M: Black hole mass in geometric units
            step_size: Integration step size (reduced for stability)
        """
        self.M = M
        self.step_size = step_size  # Reduced to 0.05 for stability
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
        k1 = geodesic_equation_accurate(y, self.M)
        k2 = geodesic_equation_accurate(y + h*k1/2, self.M)
        k3 = geodesic_equation_accurate(y + h*k2/2, self.M)
        k4 = geodesic_equation_accurate(y + h*k3, self.M)
        
        return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def integrate_geodesic(self, initial_conditions: np.ndarray, max_steps: int = 2000,
                          min_r: float = None, max_r: float = 50.0) -> Tuple[np.ndarray, bool]:
        """
        Integrate a geodesic from initial conditions. Uses Numba-optimized loop.
        """
        if min_r is None:
            min_r = 1.1 * self.rs
        traj, success = integrate_geodesic_numba(initial_conditions, self.M, self.step_size, max_steps, min_r, max_r)
        return traj, success

    def batch_integrate_geodesics(self, initial_conditions_list: List[np.ndarray], max_steps: int = 2000,
                                  min_r: float = None, max_r: float = 50.0, max_workers: int = 1) -> List[Tuple[np.ndarray, bool]]:
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
        # For light rays: gμν dxμ/dλ dxν/dλ = 0
        
        t, r, theta, phi = observer_pos
        dt, dr, dtheta, dphi = ray_direction
        
        # Get metric components at observer position
        g_tt, g_rr, g_theta_theta, g_phi_phi = schwarzschild_metric_components(r, theta, self.M)
        
        # For null geodesics: g_tt dt² + g_rr dr² + g_theta_theta dtheta² + g_phi_phi dphi² = 0
        # Since g_tt < 0, we have: |g_tt| dt² = g_rr dr² + g_theta_theta dtheta² + g_phi_phi dphi²
        
        # Normalize the spatial part first
        spatial_norm_sq = g_rr * dr*dr + g_theta_theta * dtheta*dtheta + g_phi_phi * dphi*dphi
        
        if spatial_norm_sq > 0:
            # Set dt such that null condition is satisfied
            # |g_tt| dt² = spatial_norm_sq, so dt = sqrt(spatial_norm_sq / |g_tt|)
            dt = np.sqrt(spatial_norm_sq / abs(g_tt))
        else:
            # Pure time-like ray (shouldn't happen for light)
            dt = 1.0
        
        return np.array([t, r, theta, phi, dt, dr, dtheta, dphi])

    def batch_initial_conditions_from_observer(self, observer_pos: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        """
        Fully vectorized initial conditions setup for a batch of rays.
        Args:
            observer_pos: shape (N, 4) array of [t, r, theta, phi] for each observer
            ray_direction: shape (N, 4) array of [dt, dr, dtheta, dphi] for each ray
        Returns:
            shape (N, 8) array of initial conditions for geodesic integration
        """
        t = observer_pos[:, 0]
        r = observer_pos[:, 1] 
        theta = observer_pos[:, 2]
        phi = observer_pos[:, 3]
        dt = ray_direction[:, 0]
        dr = ray_direction[:, 1]
        dtheta = ray_direction[:, 2]
        dphi = ray_direction[:, 3]
        
        # Vectorized metric components calculation
        rs = 2.0 * self.M
        r = np.maximum(r, rs + 1e-10)  # Avoid singularities
        factor = 1.0 - rs / r
        sin_theta = np.sin(theta)
        
        g_tt = -factor
        g_rr = 1.0 / factor
        g_theta_theta = r * r
        g_phi_phi = r * r * sin_theta * sin_theta
        
        # Calculate spatial norm
        spatial_norm_sq = g_rr * dr*dr + g_theta_theta * dtheta*dtheta + g_phi_phi * dphi*dphi
        
        # Set dt to satisfy null condition
        dt = np.zeros_like(dr)
        mask = spatial_norm_sq > 0
        dt[mask] = np.sqrt(spatial_norm_sq[mask] / np.abs(g_tt[mask]))
        
        initial_conditions = np.stack([t, r, theta, phi, dt, dr, dtheta, dphi], axis=1)
        return initial_conditions
