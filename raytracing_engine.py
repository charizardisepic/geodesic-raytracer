"""
Raytracing Engine for General Relativity Visualization

This module implements the core raytracing infrastructure including:
- Camera system for generating rays
- Scene management
- Ray-object intersection
- Rendering pipeline
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from geodesic_physics import GeodesicIntegrator
from numba import njit

@dataclass
class Ray:
    """Represents a light ray with origin and direction."""
    origin: np.ndarray  # [t, r, theta, phi]
    direction: np.ndarray  # [dt, dr, dtheta, dphi]

@dataclass 
class Camera:
    """Camera for generating rays from observer position."""
    position: np.ndarray  # [t, r, theta, phi] in Schwarzschild coordinates
    look_at: np.ndarray   # Target point to look at
    up_vector: np.ndarray # Up direction in local coordinates
    fov: float           # Field of view in radians
    width: int           # Image width in pixels
    height: int          # Image height in pixels
    
    def __post_init__(self):
        """Initialize camera coordinate system."""
        self.setup_coordinate_system()
    
    def setup_coordinate_system(self):
        """Set up local coordinate system for the camera."""
        # For simplicity, we'll work in a local Cartesian system
        # and convert to spherical coordinates when needed
        
        # Camera forward direction (towards black hole typically)
        forward = np.array([0, -1, 0])  # Initially pointing inward radially
        
        # Camera right and up vectors in local space
        self.right = np.array([0, 0, 1])  # phi direction
        self.up = np.array([0, 0, 0, 1])    # theta direction (modified for 4D)
        self.forward = forward
    
    def generate_ray(self, pixel_x: int, pixel_y: int) -> Ray:
        """
        Generate a ray for a given pixel.
        
        Args:
            pixel_x: X pixel coordinate (0 to width-1)
            pixel_y: Y pixel coordinate (0 to height-1)
            
        Returns:
            Ray object for the pixel
        """
        # Convert pixel coordinates to normalized device coordinates
        ndc_x = (pixel_x + 0.5) / self.width
        ndc_y = (pixel_y + 0.5) / self.height
        
        # Convert to screen space [-1, 1]
        screen_x = 2.0 * ndc_x - 1.0
        screen_y = 1.0 - 2.0 * ndc_y  # Flip Y axis
        
        # Apply aspect ratio correction
        aspect_ratio = self.width / self.height
        screen_x *= aspect_ratio
        
        # Apply field of view
        tan_half_fov = np.tan(self.fov / 2.0)
        screen_x *= tan_half_fov
        screen_y *= tan_half_fov
        
        # Convert screen coordinates to ray direction in spherical coordinates
        # This is a simplified mapping - in full GR, we'd need to account for
        # the curvature of spacetime affecting the camera's local coordinates
        
        t, r, theta, phi = self.position
        
        # Local ray direction components
        # dt component (time-like)
        dt = 1.0
        
        # Spatial components (approximate for distant observer)
        dr = -1.0  # Generally pointing toward black hole
        dtheta = screen_y * 0.1  # Small perturbation in theta
        dphi = screen_x * 0.1    # Small perturbation in phi
        
        ray_direction = np.array([dt, dr, dtheta, dphi])
        
        return Ray(origin=self.position.copy(), direction=ray_direction)

class AccretionDisk:
    """
    Simple accretion disk model for visualization.
    """
    
    def __init__(self, inner_radius: float = 3.0, outer_radius: float = 10.0,
                 thickness: float = 0.5, temperature_profile: str = "standard"):
        """
        Initialize accretion disk.
        
        Args:
            inner_radius: Inner edge of disk (typically > 3M for stable orbits)
            outer_radius: Outer edge of disk
            thickness: Disk thickness
            temperature_profile: Temperature distribution ("standard" or "custom")
        """
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.thickness = thickness
        self.temperature_profile = temperature_profile
    
    def get_emission(self, r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Get RGB emission at a given point in space.
        
        Args:
            r: Radial coordinate
            theta: Polar angle
            phi: Azimuthal angle
            
        Returns:
            RGB color tuple (0-1 range)
        """
        # Check if point is within disk
        if not self.is_in_disk(r, theta):
            return (0.0, 0.0, 0.0)
        
        # Simple temperature model: hotter closer to black hole
        if self.temperature_profile == "standard":
            # T ∝ r^(-3/4) for standard accretion disk
            normalized_r = (r - self.inner_radius) / (self.outer_radius - self.inner_radius)
            temperature = 1.0 / (0.1 + normalized_r)
        else:
            temperature = 1.0
        
        # Convert temperature to color (simplified blackbody)
        if temperature > 0.8:
            # Very hot - white/blue
            return (0.9, 0.9, 1.0)
        elif temperature > 0.5:
            # Hot - yellow/orange
            return (1.0, 0.8, 0.3)
        else:
            # Cooler - red
            return (1.0, 0.3, 0.1)
    
    def is_in_disk(self, r: float, theta: float) -> bool:
        """
        Check if a point is within the accretion disk.
        
        Args:
            r: Radial coordinate
            theta: Polar angle
            
        Returns:
            True if point is in disk
        """
        # Check radial bounds
        if r < self.inner_radius or r > self.outer_radius:
            return False
        
        # Check if close to equatorial plane
        disk_half_angle = self.thickness / r
        theta_from_equator = abs(theta - np.pi/2)
        
        return theta_from_equator < disk_half_angle

class BlackHoleRenderer:
    """
    Main rendering engine for black hole visualization.
    """
    
    def __init__(self, black_hole_mass: float = 1.0):
        """
        Initialize the renderer.
        
        Args:
            black_hole_mass: Mass of black hole in geometric units
        """
        self.black_hole_mass = black_hole_mass
        self.geodesic_integrator = GeodesicIntegrator(M=black_hole_mass)
        self.accretion_disk = AccretionDisk()
        self.background_stars = self._generate_background_stars()
    
    def _generate_background_stars(self, num_stars: int = 1000) -> List[Tuple[float, float, float]]:
        """Generate random background stars for realistic appearance."""
        stars = []
        np.random.seed(42)  # Reproducible star field
        
        for _ in range(num_stars):
            # Random position on celestial sphere
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            brightness = np.random.exponential(0.3)  # Exponential brightness distribution
            stars.append((theta, phi, min(brightness, 1.0)))
        
        return stars
    
    def trace_ray(self, ray: Ray, max_steps: int = 5000) -> Tuple[float, float, float]:
        """
        Trace a single ray through curved spacetime.
        
        Args:
            ray: Ray to trace
            max_steps: Maximum integration steps
            
        Returns:
            RGB color tuple for the ray
        """
        # Set up initial conditions for geodesic integration
        initial_conditions = self.geodesic_integrator.initial_conditions_from_observer(
            ray.origin, ray.direction
        )
        
        # Integrate the geodesic
        trajectory, escaped = self.geodesic_integrator.integrate_geodesic(
            initial_conditions, max_steps=max_steps
        )
        
        if not escaped:
            # Ray was absorbed by black hole - return black
            return (0.0, 0.0, 0.0)
        
        # Check for intersections with accretion disk
        color = self._check_disk_intersection(trajectory)
        if color is not None:
            return color
        
        # Check background stars
        final_position = trajectory[-1]
        final_theta = final_position[2]
        final_phi = final_position[3]
        
        star_color = self._sample_background_stars(final_theta, final_phi)
        return star_color
    
    def _check_disk_intersection(self, trajectory: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Check if ray intersects with accretion disk.
        
        Args:
            trajectory: Array of geodesic points
            
        Returns:
            RGB color if intersection found, None otherwise
        """
        for point in trajectory:
            t, r, theta, phi = point[:4]
            
            if self.accretion_disk.is_in_disk(r, theta):
                return self.accretion_disk.get_emission(r, theta, phi)
        
        return None
    
    def _sample_background_stars(self, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Sample background star field at given direction.
        
        Args:
            theta: Polar angle
            phi: Azimuthal angle
            
        Returns:
            RGB color from star field
        """
        # Normalize angles
        theta = theta % np.pi
        phi = phi % (2 * np.pi)
        
        # Check if close to any star
        star_radius = 0.05  # Angular radius for star visibility
        
        for star_theta, star_phi, brightness in self.background_stars:
            # Angular distance to star
            cos_dist = (np.cos(theta) * np.cos(star_theta) + 
                       np.sin(theta) * np.sin(star_theta) * np.cos(phi - star_phi))
            angular_distance = np.arccos(np.clip(cos_dist, -1, 1))
            
            if angular_distance < star_radius:
                # Star brightness based on distance to star center
                star_intensity = brightness * (1.0 - angular_distance / star_radius)
                return (star_intensity, star_intensity, star_intensity)
        
        # Deep space background
        return (0.01, 0.01, 0.02)  # Very dark blue
    
    def render_image(self, camera: Camera, filename: str = "black_hole.png") -> np.ndarray:
        """
        Render a complete image of the black hole.
        
        Args:
            camera: Camera configuration
            filename: Output filename
            
        Returns:
            RGB image array
        """
        print(f"Rendering {camera.width}x{camera.height} image...")
        
        # Initialize image array
        image = np.zeros((camera.height, camera.width, 3))
        
        # Progress bar for rendering
        total_pixels = camera.width * camera.height
        
        with tqdm(total=total_pixels, desc="Raytracing") as pbar:
            for y in range(camera.height):
                for x in range(camera.width):
                    # Generate ray for this pixel
                    ray = camera.generate_ray(x, y)
                    
                    # Trace the ray
                    color = self.trace_ray(ray)
                    
                    # Store color in image
                    image[y, x] = color
                    
                    pbar.update(1)
        
        # Save image
        self._save_image(image, filename)
        
        return image
    
    def _save_image(self, image: np.ndarray, filename: str):
        """Save rendered image to file."""
        # Convert to 8-bit RGB
        image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Save using PIL
        pil_image = Image.fromarray(image_8bit)
        pil_image.save(filename)
        print(f"Image saved as {filename}")
    
    def render_quick_preview(self, camera: Camera, scale_factor: float = 0.25) -> np.ndarray:
        """
        Render a quick low-resolution preview.
        
        Args:
            camera: Camera configuration
            scale_factor: Resolution scaling factor
            
        Returns:
            RGB image array
        """
        # Create lower resolution camera
        preview_camera = Camera(
            position=camera.position,
            look_at=camera.look_at,
            up_vector=camera.up_vector,
            fov=camera.fov,
            width=int(camera.width * scale_factor),
            height=int(camera.height * scale_factor)
        )
        
        return self.render_image(preview_camera, "black_hole_preview.png")
