#!/usr/bin/env python3
"""
Pixel-by-pixel debugging script.
Renders a tiny image and traces each pixel individually to debug the color assignment.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raytracing_engine import BlackHoleRenderer, Camera

def debug_pixel_by_pixel():
    """Debug each pixel individually to see what's happening."""
    
    # Create a tiny 5x5 image for debugging
    renderer = BlackHoleRenderer()
    
    camera = Camera(
        position=(0.0, 50.0, np.pi/2, 0.0),
        look_at=(0.0, 0.0, 0.0, 0.0),
        up_vector=(0.0, 0.0, 1.0),
        fov=np.pi/4,
        width=5,  # Very small for debugging
        height=5
    )
    
    print("=== PIXEL BY PIXEL DEBUG ===")
    print(f"Camera position: {camera.position}")
    print(f"Resolution: {camera.width}x{camera.height}")
    print()
    
    results = []
    
    for y in range(camera.height):
        for x in range(camera.width):
            print(f"PIXEL ({x}, {y}):")
            
            # Generate ray direction for this pixel (same as batch code)
            ndc_x = (x + 0.5) / camera.width
            ndc_y = (y + 0.5) / camera.height
            screen_x = 2.0 * ndc_x - 1.0
            screen_y = 1.0 - 2.0 * ndc_y
            aspect_ratio = camera.width / camera.height
            screen_x *= aspect_ratio
            tan_half_fov = np.tan(camera.fov / 2.0)
            screen_x *= tan_half_fov
            screen_y *= tan_half_fov
            
            # Ray direction components
            dt = 1.0
            dr = -1.0
            dtheta = screen_y * 0.8
            dphi = screen_x * 0.8
            
            direction = (dt, dr, dtheta, dphi)
            
            print(f"  Ray direction: {direction}")
            
            # Create initial conditions
            initial_conditions = renderer.geodesic_integrator.initial_conditions_from_observer(
                camera.position, direction
            )
            
            print(f"  Initial conditions: {initial_conditions}")
            
            # Integrate geodesic
            trajectory, escaped = renderer.geodesic_integrator.integrate_geodesic(
                initial_conditions, max_steps=2000
            )
            
            print(f"  Escaped: {escaped}")
            print(f"  Trajectory length: {len(trajectory)}")
            
            if escaped:
                final_position = trajectory[-1]
                print(f"  Final position: {final_position}")
                
                # Check for disk intersection
                disk_color = renderer._check_disk_intersection(trajectory)
                
                if disk_color is not None:
                    print(f"  DISK HIT! Color: {disk_color}")
                    color = disk_color
                else:
                    # Sample background
                    final_theta = final_position[2]
                    final_phi = final_position[3]
                    background_color = renderer._sample_background_stars(final_theta, final_phi)
                    print(f"  Background hit. Color: {background_color}")
                    color = background_color
            else:
                print(f"  BLACK HOLE HIT!")
                color = (0.0, 0.0, 0.0)
                
            # Convert to 8-bit color for comparison
            color_8bit = tuple(int(np.clip(c, 0, 1) * 255) for c in color)
            print(f"  Final color: {color} -> {color_8bit}")
            print()
            
            results.append({
                'pixel': (x, y),
                'direction': direction,
                'escaped': escaped,
                'trajectory_length': len(trajectory),
                'color': color,
                'color_8bit': color_8bit
            })
    
    # Summary
    print("=== SUMMARY ===")
    escaped_count = sum(1 for r in results if r['escaped'])
    disk_hits = sum(1 for r in results if r['escaped'] and r['color'][0] > 0.5)  # Orange-ish colors
    background_hits = sum(1 for r in results if r['escaped'] and r['color'][0] < 0.1)  # Dark colors
    
    print(f"Total pixels: {len(results)}")
    print(f"Escaped: {escaped_count} ({100*escaped_count/len(results):.1f}%)")
    print(f"Black hole hits: {len(results) - escaped_count} ({100*(len(results) - escaped_count)/len(results):.1f}%)")
    print(f"Disk hits: {disk_hits} ({100*disk_hits/len(results):.1f}%)")
    print(f"Background hits: {background_hits} ({100*background_hits/len(results):.1f}%)")
    
    # Check for uniform colors
    unique_colors = set(r['color_8bit'] for r in results)
    print(f"Unique colors: {len(unique_colors)}")
    if len(unique_colors) <= 3:
        print("Color breakdown:")
        for color in unique_colors:
            count = sum(1 for r in results if r['color_8bit'] == color)
            print(f"  {color}: {count} pixels ({100*count/len(results):.1f}%)")

if __name__ == "__main__":
    debug_pixel_by_pixel()
