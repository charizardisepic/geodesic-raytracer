#!/usr/bin/env python3
"""
Accretion Disk Diagnostic Script
Analyzes why the accretion disk is dominating the rendered image.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raytracing_engine import AccretionDisk, BlackHoleRenderer, Camera
from geodesic_physics import GeodesicIntegrator

def analyze_disk_coverage():
    """Analyze how much of the image the accretion disk covers."""
    
    # Create disk and renderer
    disk = AccretionDisk()
    renderer = BlackHoleRenderer()
    
    print("=== ACCRETION DISK PARAMETERS ===")
    print(f"Inner radius: {disk.inner_radius}")
    print(f"Outer radius: {disk.outer_radius}")
    print(f"Thickness: {disk.thickness}")
    print(f"Sample temperature at r=6: {disk.get_temperature(6.0)}")
    print(f"Sample temperature at r=8: {disk.get_temperature(8.0)}")
    print(f"Sample temperature at r=15: {disk.get_temperature(15.0)}")
    
    # Debug the temperature calculation step by step
    r = 6.0
    r_ratio = disk.inner_radius / r
    correction_factor = (1.0 - np.sqrt(r_ratio))**(1.0/4.0) if r_ratio < 1.0 else 0.0
    base_temp = (r / disk.inner_radius)**(-3.0/4.0)
    print(f"\nTemperature calculation debug for r={r}:")
    print(f"  r_ratio = {disk.inner_radius}/{r} = {r_ratio}")
    print(f"  correction_factor = {correction_factor}")
    print(f"  base_temp = {base_temp}")
    print(f"  mass_accretion_rate = {disk.mass_accretion_rate}")
    print(f"  final temp = {base_temp * correction_factor * disk.mass_accretion_rate}")
    
    # Test camera setup (same as main.py)
    camera = Camera(
        position=(0.0, 50.0, np.pi/2, 0.0),
        look_at=(0.0, 0.0, 0.0, 0.0),
        up_vector=(0.0, 0.0, 1.0),
        fov=np.pi/4,
        width=100,  # Small for testing
        height=100
    )
    
    print(f"\n=== CAMERA SETUP ===")
    print(f"Position: {camera.position}")
    print(f"Look at: {camera.look_at}")
    print(f"FOV: {camera.fov} rad ({np.degrees(camera.fov):.1f} deg)")
    print(f"Resolution: {camera.width}x{camera.height}")
    
    # Sample rays systematically and check what they hit
    hit_counts = {"black_hole": 0, "disk": 0, "background": 0}
    
    print(f"\n=== RAY SAMPLING TEST ===")
    test_rays = 50  # Sample subset for analysis
    
    for i in range(test_rays):
        for j in range(test_rays):
            # Generate ray direction
            ndc_x = (i + 0.5) / test_rays
            ndc_y = (j + 0.5) / test_rays
            screen_x = 2.0 * ndc_x - 1.0
            screen_y = 1.0 - 2.0 * ndc_y
            
            aspect_ratio = camera.width / camera.height
            screen_x *= aspect_ratio
            tan_half_fov = np.tan(camera.fov / 2.0)
            screen_x *= tan_half_fov
            screen_y *= tan_half_fov
            
            # Create ray direction
            direction = (1.0, -1.0, screen_y * 0.8, screen_x * 0.8)
            
            # Set up initial conditions
            initial_conditions = renderer.geodesic_integrator.initial_conditions_from_observer(
                camera.position, direction
            )
            
            # Integrate geodesic
            trajectory, escaped = renderer.geodesic_integrator.integrate_geodesic(
                initial_conditions, max_steps=1000
            )
            
            if not escaped:
                hit_counts["black_hole"] += 1
            else:
                # Check for disk intersection
                disk_hit = False
                for point in trajectory:
                    t, r, theta, phi = point[:4]
                    if disk.is_in_disk(r, theta):
                        hit_counts["disk"] += 1
                        disk_hit = True
                        break
                
                if not disk_hit:
                    hit_counts["background"] += 1
    
    total_rays = test_rays * test_rays
    print(f"Total rays tested: {total_rays}")
    print(f"Black hole hits: {hit_counts['black_hole']} ({100*hit_counts['black_hole']/total_rays:.1f}%)")
    print(f"Disk hits: {hit_counts['disk']} ({100*hit_counts['disk']/total_rays:.1f}%)")
    print(f"Background hits: {hit_counts['background']} ({100*hit_counts['background']/total_rays:.1f}%)")
    
    # Test disk geometry
    print(f"\n=== DISK GEOMETRY TEST ===")
    test_points = [
        (3.0, np.pi/2),      # Inner edge, equator
        (6.0, np.pi/2),      # Middle, equator
        (15.0, np.pi/2),     # Outer edge, equator
        (6.0, np.pi/2 + 0.1), # Slightly above equator
        (6.0, np.pi/2 - 0.1), # Slightly below equator
        (2.0, np.pi/2),      # Inside event horizon
        (20.0, np.pi/2),     # Outside disk
    ]
    
    for r, theta in test_points:
        in_disk = disk.is_in_disk(r, theta)
        if in_disk:
            emission = disk.get_emission(r, theta, 0.0)
            print(f"Point (r={r:.1f}, θ={theta:.3f}): IN DISK, emission={emission}")
        else:
            print(f"Point (r={r:.1f}, θ={theta:.3f}): not in disk")
    
    # Test background sampling
    print(f"\n=== BACKGROUND SAMPLING TEST ===")
    test_directions = [
        (np.pi/2, 0.0),      # Equator, front
        (np.pi/2, np.pi),    # Equator, back
        (0.0, 0.0),          # North pole
        (np.pi, 0.0),        # South pole
        (np.pi/4, np.pi/4),  # 45 degrees
    ]
    
    for theta, phi in test_directions:
        color = renderer._sample_background_stars(theta, phi)
        print(f"Direction (θ={theta:.3f}, φ={phi:.3f}): background color={color}")

def test_single_ray():
    """Test a single ray that should hit background."""
    renderer = BlackHoleRenderer()
    
    # Create a ray that should go straight out and hit background
    camera_pos = (0.0, 50.0, np.pi/2, 0.0)
    
    # Ray pointing 45 degrees up from equator - should miss disk
    direction = (1.0, -1.0, 0.5, 0.0)  # Upward direction
    
    print(f"\n=== SINGLE RAY TEST ===")
    print(f"Camera position: {camera_pos}")
    print(f"Ray direction: {direction}")
    
    # Set up initial conditions
    initial_conditions = renderer.geodesic_integrator.initial_conditions_from_observer(
        camera_pos, direction
    )
    
    print(f"Initial conditions: {initial_conditions}")
    
    # Integrate geodesic
    trajectory, escaped = renderer.geodesic_integrator.integrate_geodesic(
        initial_conditions, max_steps=1000
    )
    
    print(f"Escaped: {escaped}")
    print(f"Trajectory length: {len(trajectory)}")
    
    if escaped:
        final_position = trajectory[-1]
        print(f"Final position: {final_position}")
        
        # Check for disk intersections along path
        disk_intersections = []
        for i, point in enumerate(trajectory):
            t, r, theta, phi = point[:4]
            if renderer.accretion_disk.is_in_disk(r, theta):
                disk_intersections.append((i, point))
        
        print(f"Disk intersections: {len(disk_intersections)}")
        for i, point in disk_intersections:
            print(f"  Step {i}: {point}")
        
        if not disk_intersections:
            # Sample background
            final_theta = final_position[2]
            final_phi = final_position[3]
            background_color = renderer._sample_background_stars(final_theta, final_phi)
            print(f"Background color: {background_color}")

def main():
    """Main diagnostic."""
    print("ACCRETION DISK DIAGNOSTIC")
    print("=" * 40)
    
    analyze_disk_coverage()
    test_single_ray()

if __name__ == "__main__":
    main()
