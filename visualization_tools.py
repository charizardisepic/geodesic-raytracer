"""
Visualization utilities for analyzing geodesics and spacetime curvature.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geodesic_physics import GeodesicIntegrator
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GeodesicVisualizer:
    """
    Utility class for visualizing geodesic paths and spacetime properties.
    """
    
    def __init__(self, black_hole_mass: float = 1.0):
        """
        Initialize the visualizer.
        
        Args:
            black_hole_mass: Mass of the black hole
        """
        self.M = black_hole_mass
        self.rs = 2.0 * black_hole_mass
        self.integrator = GeodesicIntegrator(M=black_hole_mass)
    
    def plot_geodesic_3d(self, initial_conditions: np.ndarray, max_steps: int = 5000, 
                        title: str = "Geodesic Path") -> plt.Figure:
        """
        Plot a 3D visualization of a geodesic path.
        
        Args:
            initial_conditions: Initial conditions for the geodesic
            max_steps: Maximum integration steps
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Integrate geodesic
        trajectory, escaped = self.integrator.integrate_geodesic(
            initial_conditions, max_steps=max_steps
        )
        
        # Convert to Cartesian coordinates for visualization
        x_coords = []
        y_coords = []
        z_coords = []
        
        for point in trajectory:
            t, r, theta, phi = point[:4]
            
            # Convert spherical to Cartesian
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot geodesic
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, alpha=0.8, label='Geodesic')
        
        # Plot black hole (event horizon)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = self.rs * np.outer(np.cos(u), np.sin(v))
        y_sphere = self.rs * np.outer(np.sin(u), np.sin(v))
        z_sphere = self.rs * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='black')
        
        # Mark start and end points
        if len(x_coords) > 0:
            ax.scatter([x_coords[0]], [y_coords[0]], [z_coords[0]], 
                      c='green', s=100, label='Start')
            ax.scatter([x_coords[-1]], [y_coords[-1]], [z_coords[-1]], 
                      c='red', s=100, label='End')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        # Set equal aspect ratio
        max_range = max([max(x_coords) - min(x_coords),
                        max(y_coords) - min(y_coords),
                        max(z_coords) - min(z_coords)]) / 2.0
        mid_x = (max(x_coords) + min(x_coords)) * 0.5
        mid_y = (max(y_coords) + min(y_coords)) * 0.5
        mid_z = (max(z_coords) + min(z_coords)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        return fig
    
    def plot_effective_potential(self, angular_momentum: float, 
                               energy: float = 1.0) -> plt.Figure:
        """
        Plot the effective potential for geodesics.
        
        Args:
            angular_momentum: Angular momentum parameter
            energy: Energy parameter
            
        Returns:
            Matplotlib figure
        """
        r_values = np.linspace(1.5 * self.rs, 20 * self.rs, 1000)
        
        # Effective potential for massive particles
        # V_eff = (1 - rs/r)(1 + L²/r²)
        L = angular_momentum
        rs = self.rs
        
        V_eff = []
        for r in r_values:
            if r > rs:
                potential = (1 - rs/r) * (1 + L*L/(r*r))
                V_eff.append(potential)
            else:
                V_eff.append(float('inf'))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(r_values / self.rs, V_eff, 'b-', linewidth=2, label=f'L = {L:.2f}')
        ax.axhline(y=energy, color='r', linestyle='--', linewidth=2, label=f'E = {energy:.2f}')
        ax.axvline(x=1.0, color='k', linestyle='-', alpha=0.5, label='Event Horizon')
        
        ax.set_xlabel('r / rs')
        ax.set_ylabel('Effective Potential')
        ax.set_title('Effective Potential for Geodesic Motion')
        ax.set_ylim(0, 2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def analyze_photon_sphere(self) -> Tuple[float, plt.Figure]:
        """
        Analyze the photon sphere and plot circular photon orbits.
        
        Returns:
            Tuple of (photon sphere radius, plot figure)
        """
        # Photon sphere radius for Schwarzschild black hole
        r_photon = 1.5 * self.rs
        
        print(f"Photon sphere radius: {r_photon:.3f} (1.5 * rs)")
        print(f"Event horizon radius: {self.rs:.3f}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Radial cross-section showing key radii
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Event horizon
        x_eh = self.rs * np.cos(theta)
        y_eh = self.rs * np.sin(theta)
        ax1.fill(x_eh, y_eh, color='black', alpha=0.8, label='Event Horizon')
        
        # Photon sphere
        x_ps = r_photon * np.cos(theta)
        y_ps = r_photon * np.sin(theta)
        ax1.plot(x_ps, y_ps, 'r--', linewidth=2, label='Photon Sphere')
        
        # ISCO (Innermost Stable Circular Orbit) for reference
        r_isco = 3 * self.rs
        x_isco = r_isco * np.cos(theta)
        y_isco = r_isco * np.sin(theta)
        ax1.plot(x_isco, y_isco, 'g:', linewidth=2, label='ISCO')
        
        ax1.set_xlim(-4*self.rs, 4*self.rs)
        ax1.set_ylim(-4*self.rs, 4*self.rs)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x / M')
        ax1.set_ylabel('y / M')
        ax1.set_title('Black Hole Structure (Equatorial Plane)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Light ray bending for different impact parameters
        impact_parameters = [r_photon * 0.9, r_photon, r_photon * 1.1, r_photon * 1.5]
        colors = ['red', 'orange', 'blue', 'green']
        
        for i, b in enumerate(impact_parameters):
            # Initial conditions for light ray
            r_initial = 20 * self.rs  # Start far away
            theta_initial = np.pi / 2  # Equatorial plane
            phi_initial = 0
            
            # For light ray with impact parameter b
            # E = 1 (can set energy to 1 for light)
            # L = b (angular momentum equals impact parameter for light)
            dr_initial = -1.0  # Moving inward
            dtheta_initial = 0.0  # Stay in equatorial plane
            dphi_initial = b / (r_initial * r_initial)  # From angular momentum
            dt_initial = 1.0
            
            initial_conditions = np.array([
                0, r_initial, theta_initial, phi_initial,
                dt_initial, dr_initial, dtheta_initial, dphi_initial
            ])
            
            trajectory, escaped = self.integrator.integrate_geodesic(
                initial_conditions, max_steps=3000
            )
            
            # Convert to Cartesian and plot
            x_coords = []
            y_coords = []
            
            for point in trajectory:
                t, r, theta, phi = point[:4]
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                x_coords.append(x)
                y_coords.append(y)
            
            label = f'b = {b/self.rs:.2f} rs'
            if b == r_photon:
                label += ' (Critical)'
            
            ax2.plot(x_coords, y_coords, color=colors[i], linewidth=2, 
                    label=label, alpha=0.8)
        
        # Add black hole visualization
        ax2.fill(x_eh, y_eh, color='black', alpha=0.8)
        ax2.plot(x_ps, y_ps, 'r--', linewidth=1, alpha=0.5)
        
        ax2.set_xlim(-10*self.rs, 10*self.rs)
        ax2.set_ylim(-10*self.rs, 10*self.rs)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x / M')
        ax2.set_ylabel('y / M')
        ax2.set_title('Light Ray Trajectories')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return r_photon, fig
    
    def plot_light_ray_deflection(self, impact_parameters: List[float]) -> plt.Figure:
        """
        Plot light ray deflection for different impact parameters.
        
        Args:
            impact_parameters: List of impact parameters to analyze
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        deflection_angles = []
        valid_b_values = []
        
        for b in impact_parameters:
            if b <= 1.5 * self.rs:
                # Too close - ray will be captured
                continue
                
            # Set up initial conditions
            r_initial = 50 * self.rs
            initial_conditions = np.array([
                0, r_initial, np.pi/2, 0,  # position
                1.0, -1.0, 0.0, b/(r_initial*r_initial)  # velocity
            ])
            
            trajectory, escaped = self.integrator.integrate_geodesic(
                initial_conditions, max_steps=5000
            )
            
            if escaped and len(trajectory) > 10:
                # Calculate deflection angle
                initial_phi = trajectory[0][3]
                final_phi = trajectory[-1][3]
                deflection = abs(final_phi - initial_phi)
                
                # For small angles, deflection ≈ 4M/b (classical prediction)
                deflection_angles.append(deflection)
                valid_b_values.append(b)
        
        # Plot results
        ax.loglog(valid_b_values, deflection_angles, 'bo-', linewidth=2, 
                 markersize=6, label='Numerical Integration')
        
        # Classical prediction for comparison
        b_classical = np.array(valid_b_values)
        deflection_classical = 4 * self.M / b_classical
        ax.loglog(b_classical, deflection_classical, 'r--', linewidth=2, 
                 label='Classical Prediction (4M/b)')
        
        ax.set_xlabel('Impact Parameter (b/rs)')
        ax.set_ylabel('Deflection Angle (radians)')
        ax.set_title('Light Ray Deflection vs Impact Parameter')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
