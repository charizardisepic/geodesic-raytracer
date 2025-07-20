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
import os
import multiprocessing

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
        # Camera forward direction (towards black hole typically)
        forward = np.array([0, -1, 0])  # Initially pointing inward radially
        
        # Camera right and up vectors in local space
        self.right = np.array([0, 0, 1])  # phi direction
        self.up = np.array([0, 0, 0, 1])    # theta direction (modified for 4D)
        self.forward = forward
    
    def generate_ray(self, pixel_x: int, pixel_y: int) -> Ray:
        """
        Generate a ray for a given pixel with perfect coordinate symmetry.
        
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
        t, r, theta, phi = self.position
        
        # Local ray direction components
        dt = 1.0
        dr = -1.0  # Generally pointing toward black hole
        dtheta = screen_y * 0.8  # Angular range for proper impact parameters
        dphi = screen_x * 0.8    # Angular range for proper impact parameters
        
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
        theta_from_equator = np.abs(theta - np.pi/2)
        
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
        
        # Warm up Numba JIT compilation for immediate performance
        self._warmup_numba()
    
    def _warmup_numba(self):
        """Pre-compile Numba functions for immediate fast performance."""
        # Create a dummy initial condition to trigger Numba compilation
        dummy_ic = np.array([0.0, 20.0, np.pi/2, 0.0, 1.0, -1.0, 0.0, 0.1])
        # Run a single fast integration to compile functions
        from geodesic_physics import integrate_geodesic_numba
        integrate_geodesic_numba(dummy_ic, self.black_hole_mass, 0.5, 10, 2.1, 30.0)

    def _generate_background_stars(self, num_stars: int = 0) -> List[Tuple[float, float, float]]:
        """Generate random background stars for realistic appearance."""
        # For simple scene demo, disable stars (white background only)
        return []
    
    def trace_ray(self, ray: Ray, max_steps: int = 2000) -> Tuple[float, float, float]:
        """
        Trace a single ray through curved spacetime.
        
        Args:
            ray: Ray to trace
            max_steps: Maximum integration steps (reduced for speed)
            
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
        
        # Check for intersections with accretion disk (disabled for simple scene)
        # color = self._check_disk_intersection(trajectory)
        # if color is not None:
        #     return color
        
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
        Sample background starscape at given direction.
        
        Args:
            theta: Polar angle
            phi: Azimuthal angle
            
        Returns:
            RGB color from starscape
        """
        theta = theta % np.pi
        phi = phi % (2 * np.pi)
        
        # Create a pseudo-random starfield using deterministic noise
        # Use spherical coordinates as seed for consistent star placement
        
        # Convert spherical coordinates to deterministic star positions
        coord_hash = np.sin(theta * 37.3) * np.cos(phi * 23.7) + np.sin(phi * 41.1) * np.cos(theta * 19.4)
        coord_hash = (coord_hash + 1.0) / 2.0  # Normalize to [0, 1]
        
        # Create multiple octaves of star noise for realistic distribution
        star_density = 0.01  # Increased probability for better visibility of distortion
        
        # Generate star positions using multiple hash functions
        hash1 = abs(np.sin(theta * 127.1 + phi * 311.7) * 43758.5453)
        hash2 = abs(np.sin(theta * 269.5 + phi * 183.3) * 43758.5453)
        hash3 = abs(np.sin(theta * 419.2 + phi * 371.9) * 43758.5453)
        
        # Add some bright reference stars for clear distortion visibility
        # Create a few bright stars at specific locations
        bright_star_hash = abs(np.sin(theta * 73.7 + phi * 157.3) * 43758.5453)
        bright_star_hash = bright_star_hash - int(bright_star_hash)
        
        if bright_star_hash < 0.005:  # Bright reference stars
            brightness = 0.8 + 0.2 * hash2
            return (brightness, brightness * 0.9, brightness * 0.7)  # Bright yellow-white
        
        # Extract fractional parts for randomness
        hash1 = hash1 - int(hash1)
        hash2 = hash2 - int(hash2)
        hash3 = hash3 - int(hash3)
        
        # Check if we have a star at this location
        if hash1 < star_density:
            # Star brightness varies
            brightness = 0.3 + 0.7 * hash2
            
            # Star color varies (blue-white to yellow-white)
            if hash3 < 0.1:  # Blue giants (rare)
                return (brightness * 0.7, brightness * 0.8, brightness * 1.0)
            elif hash3 < 0.3:  # White stars
                return (brightness, brightness, brightness)
            elif hash3 < 0.7:  # Yellow-white stars
                return (brightness * 1.0, brightness * 0.95, brightness * 0.8)
            else:  # Red giants
                return (brightness * 1.0, brightness * 0.7, brightness * 0.5)
        
        # Background space color - very dark blue/black
        return (0.02, 0.02, 0.05)
    
    def render_image(self, camera: Camera, filename: str = "black_hole.png", batch_size: int = 250, max_workers: int = 1) -> np.ndarray:
        """
        Render a complete image of the black hole using batch geodesic integration with progress bar.
        """
        print(f"Rendering {camera.width}x{camera.height} image (batch mode)...")
        image = np.zeros((camera.height, camera.width, 3))
        total_pixels = camera.width * camera.height
        
        # Vectorized ray preparation with FIXED symmetric coordinates
        xs, ys = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
        xs = xs.flatten()
        ys = ys.flatten()
        
        # Precompute all ray directions in a single batch
        ndc_x = (xs + 0.5) / camera.width
        ndc_y = (ys + 0.5) / camera.height
        screen_x = 2.0 * ndc_x - 1.0
        screen_y = 1.0 - 2.0 * ndc_y
        aspect_ratio = camera.width / camera.height
        screen_x *= aspect_ratio
        tan_half_fov = np.tan(camera.fov / 2.0)
        screen_x *= tan_half_fov
        screen_y *= tan_half_fov
        
        t, r, theta, phi = camera.position
        dt = np.ones_like(xs)
        dr = -np.ones_like(xs)
        dtheta = screen_y * 0.8
        dphi = screen_x * 0.8
        
        ray_origins = np.tile(camera.position, (total_pixels, 1))
        ray_directions = np.stack([dt, dr, dtheta, dphi], axis=1)
        
        # Fast vectorized initial conditions (no loops)
        print("Preparing initial conditions...")
        initial_conditions_array = self.geodesic_integrator.batch_initial_conditions_from_observer(ray_origins, ray_directions)
        pixel_indices = list(zip(ys, xs))
        
        print("Tracing rays in parallel...")
        from tqdm import tqdm
        with tqdm(total=total_pixels, desc="Raytracing") as pbar:
            # Process in larger batches for efficiency
            for batch_start in range(0, total_pixels, batch_size):
                batch_end = min(batch_start + batch_size, total_pixels)
                batch_ics = [initial_conditions_array[i] for i in range(batch_start, batch_end)]
                batch_indices = pixel_indices[batch_start:batch_end]
                
                # Use appropriate integration settings for accuracy
                batch_results = self.geodesic_integrator.batch_integrate_geodesics(
                    batch_ics, max_steps=2000, max_workers=max_workers
                )
                
                for i, (trajectory, escaped) in enumerate(batch_results):
                    y, x = batch_indices[i]
                    if not escaped:
                        color = (0.0, 0.0, 0.0)  # Black hole
                    else:
                        # Skip disk intersection for simple scene performance
                        # color = self._check_disk_intersection(trajectory)
                        # if color is None:
                        final_position = trajectory[-1]
                        final_theta = final_position[2]
                        final_phi = final_position[3]
                        color = self._sample_background_stars(final_theta, final_phi)
                    image[y, x] = color
                    pbar.update(1)
        
        self._save_image(image, filename)
        return image
    
    def _save_image(self, image: np.ndarray, filename: str):
        """Save rendered image to file."""
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
        
        return self.render_image(preview_camera, "output/output.png")
