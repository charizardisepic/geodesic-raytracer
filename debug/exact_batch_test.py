#!/usr/bin/env python3
"""
Exact Batch Reproduction Test
Renders the exact same 5x5 grid through batch processing and compares to individual results.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raytracing_engine import BlackHoleRenderer, Camera

def test_exact_batch_reproduction():
    """Test the exact same 5x5 grid through batch processing."""
    
    renderer = BlackHoleRenderer()
    
    camera = Camera(
        position=(0.0, 50.0, np.pi/2, 0.0),
        look_at=(0.0, 0.0, 0.0, 0.0),
        up_vector=(0.0, 0.0, 1.0),
        fov=np.pi/4,
        width=5,
        height=5
    )
    
    print("=== EXACT BATCH REPRODUCTION TEST ===")
    print(f"Testing {camera.width}x{camera.height} grid")
    print()
    
    # Render through the batch system
    print("Rendering through batch system...")
    image = renderer.render_image(camera, "debug/test_5x5.png", batch_size=25, max_workers=1)
    
    print(f"Batch image shape: {image.shape}")
    print()
    
    # Convert to 8-bit for comparison
    image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Expected results from individual test:
    # 24 pixels: (12, 12, 38) - background
    # 1 pixel (2,2): (0, 0, 0) - black hole
    
    expected_background = (12, 12, 38)
    expected_black_hole = (0, 0, 0)
    
    print("PIXEL BY PIXEL COMPARISON:")
    background_count = 0
    black_hole_count = 0
    other_colors = []
    
    for y in range(camera.height):
        for x in range(camera.width):
            pixel_color = tuple(image_8bit[y, x])
            
            if pixel_color == expected_black_hole:
                print(f"Pixel ({x},{y}): {pixel_color} - BLACK HOLE ✓")
                black_hole_count += 1
            elif pixel_color == expected_background:
                print(f"Pixel ({x},{y}): {pixel_color} - BACKGROUND ✓")
                background_count += 1
            else:
                print(f"Pixel ({x},{y}): {pixel_color} - UNEXPECTED ✗")
                other_colors.append(pixel_color)
    
    print()
    print("SUMMARY:")
    print(f"Background pixels: {background_count} (expected: 24)")
    print(f"Black hole pixels: {black_hole_count} (expected: 1)")
    print(f"Unexpected pixels: {len(other_colors)}")
    
    if other_colors:
        print("Unexpected colors:")
        for color in set(other_colors):
            count = other_colors.count(color)
            print(f"  {color}: {count} pixels")
    
    # Check if results match expectation
    if background_count == 24 and black_hole_count == 1 and len(other_colors) == 0:
        print("\n✅ SUCCESS: Batch results match individual results!")
    else:
        print("\n❌ FAILURE: Batch results don't match individual results!")
        
        # Analyze the discrepancy
        print("\nDISCREPANCY ANALYSIS:")
        
        # Check unique colors in batch result
        unique_colors = []
        for y in range(camera.height):
            for x in range(camera.width):
                color = tuple(image_8bit[y, x])
                if color not in unique_colors:
                    unique_colors.append(color)
        
        print(f"Unique colors in batch result: {len(unique_colors)}")
        for color in unique_colors:
            count = sum(1 for y in range(camera.height) for x in range(camera.width) 
                       if tuple(image_8bit[y, x]) == color)
            print(f"  {color}: {count} pixels ({100*count/25:.1f}%)")

if __name__ == "__main__":
    test_exact_batch_reproduction()
