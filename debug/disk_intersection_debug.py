#!/usr/bin/env python3
"""
Disk Intersection Debug
Debug the _check_disk_intersection function to see why it might be returning disk colors for background rays.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raytracing_engine import BlackHoleRenderer, Camera

def debug_disk_intersection():
    """Debug the disk intersection logic."""
    
    renderer = BlackHoleRenderer()
    
    # Test a background ray from the pixel debug
    camera_pos = (0.0, 50.0, np.pi/2, 0.0)
    direction = (1.0, -1.0, 0.26509667991878083, -0.26509667991878083)  # From pixel (0,0)
    
    print("=== DISK INTERSECTION DEBUG ===")
    print(f"Camera position: {camera_pos}")
    print(f"Ray direction: {direction}")
    print()
    
    # Create initial conditions and integrate
    initial_conditions = renderer.geodesic_integrator.initial_conditions_from_observer(
        camera_pos, direction
    )
    
    trajectory, escaped = renderer.geodesic_integrator.integrate_geodesic(
        initial_conditions, max_steps=2000
    )
    
    print(f"Escaped: {escaped}")
    print(f"Trajectory length: {len(trajectory)}")
    print()
    
    if escaped and len(trajectory) > 0:
        # Debug the disk intersection check step by step
        print("DISK INTERSECTION CHECK:")
        print(f"Disk inner radius: {renderer.accretion_disk.inner_radius}")
        print(f"Disk outer radius: {renderer.accretion_disk.outer_radius}")
        print(f"Disk thickness: {renderer.accretion_disk.thickness}")
        print()
        
        disk_hits = []
        for i, point in enumerate(trajectory):
            t, r, theta, phi = point[:4]
            
            # Check if this point is in disk
            in_disk = renderer.accretion_disk.is_in_disk(r, theta)
            
            if in_disk:
                disk_hits.append((i, point, r, theta))
                print(f"Step {i}: DISK HIT at r={r:.3f}, θ={theta:.3f}")
                
                # Get emission for this point
                emission = renderer.accretion_disk.get_emission(r, theta, phi)
                print(f"  Emission: {emission}")
        
        print(f"\nTotal disk hits: {len(disk_hits)}")
        
        # Now call the actual _check_disk_intersection function
        disk_color = renderer._check_disk_intersection(trajectory)
        print(f"_check_disk_intersection result: {disk_color}")
        
        if disk_color is not None:
            print("ERROR: This ray should hit background, not disk!")
        else:
            print("Correct: Ray hits background")
            
            # Test background sampling
            final_position = trajectory[-1]
            final_theta = final_position[2]
            final_phi = final_position[3]
            background_color = renderer._sample_background_stars(final_theta, final_phi)
            print(f"Background color: {background_color}")
        
        print()
        print("TRAJECTORY SAMPLE (first 5 and last 5 points):")
        for i in range(min(5, len(trajectory))):
            point = trajectory[i]
            t, r, theta, phi = point[:4]
            print(f"  Step {i}: t={t:.3f}, r={r:.3f}, θ={theta:.3f}, φ={phi:.3f}")
        
        if len(trajectory) > 10:
            print("  ...")
            for i in range(max(5, len(trajectory)-5), len(trajectory)):
                point = trajectory[i]
                t, r, theta, phi = point[:4]
                print(f"  Step {i}: t={t:.3f}, r={r:.3f}, θ={theta:.3f}, φ={phi:.3f}")

def test_disk_intersection_with_known_disk_ray():
    """Test with a ray that should definitely hit the disk."""
    
    renderer = BlackHoleRenderer()
    
    print("\n=== TESTING WITH DISK RAY ===")
    
    # Create a ray that goes straight through the equatorial plane
    camera_pos = (0.0, 50.0, np.pi/2, 0.0)
    direction = (1.0, -1.0, 0.0, 0.0)  # Straight inward, equatorial
    
    print(f"Camera position: {camera_pos}")
    print(f"Ray direction: {direction}")
    print()
    
    initial_conditions = renderer.geodesic_integrator.initial_conditions_from_observer(
        camera_pos, direction
    )
    
    trajectory, escaped = renderer.geodesic_integrator.integrate_geodesic(
        initial_conditions, max_steps=2000
    )
    
    print(f"Escaped: {escaped}")
    print(f"Trajectory length: {len(trajectory)}")
    
    if not escaped:
        print("Ray was absorbed by black hole, checking disk intersections along path:")
        
        disk_hits = []
        for i, point in enumerate(trajectory):
            t, r, theta, phi = point[:4]
            
            if renderer.accretion_disk.is_in_disk(r, theta):
                disk_hits.append((i, r, theta))
                if len(disk_hits) <= 3:  # Show first few hits
                    emission = renderer.accretion_disk.get_emission(r, theta, phi)
                    print(f"  Step {i}: r={r:.3f}, θ={theta:.3f}, emission={emission}")
        
        print(f"Total disk hits: {len(disk_hits)}")
        
        disk_color = renderer._check_disk_intersection(trajectory)
        print(f"_check_disk_intersection result: {disk_color}")

if __name__ == "__main__":
    debug_disk_intersection()
    test_disk_intersection_with_known_disk_ray()
