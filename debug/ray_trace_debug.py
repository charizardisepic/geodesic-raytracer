#!/usr/bin/env python3
"""
Ray Tracing Diagnostic Script

Debug the ray tracing pipeline to understand why only uniform blue background
is being rendered instead of stars and accretion disk.
"""

import numpy as np
import sys
import os
sys.path.append('.')

from geodesic_physics import GeodesicIntegrator
from raytracing_engine import BlackHoleRenderer, Camera

def debug_single_ray(renderer, camera, pixel_x, pixel_y):
    """
    Debug a single ray to see what happens during tracing.
    
    Args:
        renderer: BlackHoleRenderer instance
        camera: Camera instance
        pixel_x, pixel_y: Pixel coordinates to trace
    """
    print(f"Debugging ray for pixel ({pixel_x}, {pixel_y})")
    print("=" * 50)
    
    # Generate ray
    ray = camera.generate_ray(pixel_x, pixel_y)
    print(f"Ray origin: {ray.origin}")
    print(f"Ray direction: {ray.direction}")
    
    # Set up initial conditions
    initial_conditions = renderer.geodesic_integrator.initial_conditions_from_observer(
        ray.origin, ray.direction
    )
    print(f"Initial conditions: {initial_conditions}")
    
    # Integrate geodesic with detailed tracking
    print("\nIntegrating geodesic...")
    trajectory, escaped = renderer.geodesic_integrator.integrate_geodesic(
        initial_conditions, max_steps=2000
    )
    
    print(f"Trajectory length: {len(trajectory)}")
    print(f"Ray escaped: {escaped}")
    
    if len(trajectory) > 0:
        print(f"First point: {trajectory[0]}")
        print(f"Last point: {trajectory[-1]}")
        
        # Check minimum radius reached
        radii = trajectory[:, 1]
        min_radius = np.min(radii)
        print(f"Minimum radius reached: {min_radius:.3f}")
        print(f"Event horizon (r=2): {min_radius < 2.0}")
    
    if not escaped:
        print("Ray was absorbed by black hole")
        return (0.0, 0.0, 0.0)
    
    # Check disk intersection
    print("\nChecking disk intersection...")
    disk_color = renderer._check_disk_intersection(trajectory)
    if disk_color is not None:
        print(f"Disk intersection found! Color: {disk_color}")
        return disk_color
    else:
        print("No disk intersection")
    
    # Check background sampling
    print("\nSampling background...")
    final_position = trajectory[-1]
    final_theta = final_position[2]
    final_phi = final_position[3]
    
    print(f"Final position: theta={final_theta:.3f}, phi={final_phi:.3f}")
    
    # Debug the background sampling function
    star_color = debug_background_sampling(final_theta, final_phi)
    print(f"Background color: {star_color}")
    
    return star_color

def debug_background_sampling(theta, phi):
    """Debug the background sampling function."""
    print(f"  Input: theta={theta:.3f}, phi={phi:.3f}")
    
    theta = theta % np.pi
    phi = phi % (2 * np.pi)
    print(f"  Normalized: theta={theta:.3f}, phi={phi:.3f}")
    
    # Star density check
    star_density = 0.01
    hash1 = abs(np.sin(theta * 127.1 + phi * 311.7) * 43758.5453)
    hash1 = hash1 - int(hash1)
    print(f"  Hash1: {hash1:.6f}, threshold: {star_density}")
    
    # Bright star check
    bright_star_hash = abs(np.sin(theta * 73.7 + phi * 157.3) * 43758.5453)
    bright_star_hash = bright_star_hash - int(bright_star_hash)
    print(f"  Bright star hash: {bright_star_hash:.6f}, threshold: 0.005")
    
    if bright_star_hash < 0.005:
        print("  -> BRIGHT STAR HIT!")
        return (0.9, 0.81, 0.63)
    
    if hash1 < star_density:
        print("  -> NORMAL STAR HIT!")
        return (0.7, 0.7, 0.7)
    
    print("  -> SPACE BACKGROUND")
    return (0.02, 0.02, 0.05)

def debug_disk_intersection(renderer):
    """Debug the accretion disk intersection logic."""
    print("\nDebugging Accretion Disk:")
    print("=" * 50)
    
    disk = renderer.accretion_disk
    print(f"Inner radius: {disk.inner_radius}")
    print(f"Outer radius: {disk.outer_radius}")
    print(f"Thickness: {disk.thickness}")
    
    # Test some points in the disk
    test_points = [
        (8.0, np.pi/2, 0.0),    # In disk, mid-plane
        (8.0, np.pi/2 + 0.05, 0.0),  # Slightly above disk
        (3.0, np.pi/2, 0.0),    # Too close to black hole
        (100.0, np.pi/2, 0.0),  # Too far from disk
    ]
    
    for r, theta, phi in test_points:
        in_disk = disk.is_in_disk(r, theta)
        if in_disk:
            color = disk.get_emission(r, theta, phi)
            print(f"Point r={r}, θ={theta:.3f}: IN DISK, color={color}")
        else:
            print(f"Point r={r}, θ={theta:.3f}: NOT in disk")

def main():
    """Main diagnostic function."""
    print("Ray Tracing Diagnostic")
    print("=" * 60)
    
    # Set up renderer and camera (same as main render)
    renderer = BlackHoleRenderer(black_hole_mass=1.0)
    
    observer_position = np.array([0.0, 8.0, np.pi/2, 0.0])
    look_at_point = np.array([0.0, 2.5, np.pi/2, 0.0])
    up_vector = np.array([0.0, 0.0, 1.0])
    
    camera = Camera(
        position=observer_position,
        look_at=look_at_point,
        up_vector=up_vector,
        fov=np.pi/3,
        width=400,
        height=300
    )
    
    print(f"Camera position: {camera.position}")
    print(f"Image dimensions: {camera.width}x{camera.height}")
    
    # Debug the accretion disk
    debug_disk_intersection(renderer)
    
    # Test several different rays
    test_pixels = [
        (200, 150),  # Center
        (100, 75),   # Upper left
        (300, 225),  # Lower right
        (50, 150),   # Left edge
        (350, 150),  # Right edge
    ]
    
    for px, py in test_pixels:
        print("\n" + "="*80)
        color = debug_single_ray(renderer, camera, px, py)
        print(f"FINAL COLOR: {color}")
    
    # Quick test of background function directly
    print("\n" + "="*80)
    print("Direct Background Function Test:")
    test_angles = [
        (0.0, 0.0),
        (np.pi/4, np.pi/4),
        (np.pi/2, np.pi),
        (np.pi, 2*np.pi),
    ]
    
    for theta, phi in test_angles:
        color = debug_background_sampling(theta, phi)
        print(f"θ={theta:.3f}, φ={phi:.3f} -> {color}")

if __name__ == "__main__":
    main()
