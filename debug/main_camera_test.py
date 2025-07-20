#!/usr/bin/env python3
"""
Main.py Camera Test
Test with the exact same camera setup as main.py to see if most rays hit the disk.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raytracing_engine import BlackHoleRenderer, Camera

def test_main_camera_setup():
    """Test with the exact same camera setup as main.py."""
    
    renderer = BlackHoleRenderer(black_hole_mass=1.0)
    
    # EXACT same setup as main.py
    observer_distance = 8.0  # Much closer!
    observer_position = np.array([0.0, observer_distance, np.pi/2, 0.0])
    look_at_point = np.array([0.0, 2.5, np.pi/2, 0.0])  # Looking at r=2.5
    up_vector = np.array([0.0, 0.0, 1.0])
    
    camera = Camera(
        position=observer_position,
        look_at=look_at_point,
        up_vector=up_vector,
        fov=np.pi/3,  # 60 degrees
        width=10,  # Small test
        height=10
    )
    
    print("=== MAIN.PY CAMERA TEST ===")
    print(f"Observer position: {observer_position}")
    print(f"Observer distance: {observer_distance}")
    print(f"Look at point: {look_at_point}")
    print(f"FOV: {camera.fov} rad ({np.degrees(camera.fov):.1f} deg)")
    print(f"Disk inner radius: {renderer.accretion_disk.inner_radius}")
    print(f"Disk outer radius: {renderer.accretion_disk.outer_radius}")
    print()
    
    print("ANALYSIS:")
    print(f"Observer at r={observer_distance}, disk spans r={renderer.accretion_disk.inner_radius} to r={renderer.accretion_disk.outer_radius}")
    if observer_distance < renderer.accretion_disk.outer_radius:
        print(f"⚠️  OBSERVER IS INSIDE DISK OUTER RADIUS!")
    if observer_distance > renderer.accretion_disk.inner_radius:
        print(f"Observer is outside disk inner radius (good)")
    print()
    
    # Sample a few rays to see what they hit
    print("SAMPLING RAYS:")
    hit_counts = {"black_hole": 0, "disk": 0, "background": 0}
    sample_size = 25  # 5x5 sampling
    
    for i in range(5):
        for j in range(5):
            pixel_x, pixel_y = i, j
            
            # Generate ray direction (same as batch code)
            ndc_x = (pixel_x + 0.5) / 5
            ndc_y = (pixel_y + 0.5) / 5
            screen_x = 2.0 * ndc_x - 1.0
            screen_y = 1.0 - 2.0 * ndc_y
            aspect_ratio = 1.0  # 5x5 is square
            screen_x *= aspect_ratio
            tan_half_fov = np.tan(camera.fov / 2.0)
            screen_x *= tan_half_fov
            screen_y *= tan_half_fov
            
            dt = 1.0
            dr = -1.0
            dtheta = screen_y * 0.8
            dphi = screen_x * 0.8
            
            direction = (dt, dr, dtheta, dphi)
            
            # Trace ray
            initial_conditions = renderer.geodesic_integrator.initial_conditions_from_observer(
                camera.position, direction
            )
            
            trajectory, escaped = renderer.geodesic_integrator.integrate_geodesic(
                initial_conditions, max_steps=2000
            )
            
            if not escaped:
                hit_counts["black_hole"] += 1
                result = "BLACK_HOLE"
            else:
                disk_color = renderer._check_disk_intersection(trajectory)
                if disk_color is not None:
                    hit_counts["disk"] += 1
                    result = "DISK"
                else:
                    hit_counts["background"] += 1
                    result = "BACKGROUND"
            
            if i < 2 and j < 2:  # Show details for first few rays
                print(f"  Pixel ({pixel_x},{pixel_y}): {result}")
    
    print()
    print("RESULTS:")
    total = sum(hit_counts.values())
    for hit_type, count in hit_counts.items():
        print(f"{hit_type}: {count}/{total} ({100*count/total:.1f}%)")
    
    # Now render the actual small image
    print("\nRendering 10x10 image with main.py camera...")
    image = renderer.render_image(camera, "debug/main_camera_test.png", batch_size=100, max_workers=1)
    
    # Convert to 8-bit and analyze
    image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Count unique colors
    unique_colors = {}
    for y in range(10):
        for x in range(10):
            color = tuple(image_8bit[y, x])
            unique_colors[color] = unique_colors.get(color, 0) + 1
    
    print("\nBATCH RENDER RESULTS:")
    print(f"Unique colors: {len(unique_colors)}")
    for color, count in unique_colors.items():
        print(f"  {color}: {count} pixels ({100*count/100:.1f}%)")

if __name__ == "__main__":
    test_main_camera_setup()
