#!/usr/bin/env python3
"""
Main script for rendering black hole images using general relativity raytracing.

This script demonstrates the complete pipeline:
1. Set up camera and scene
2. Initialize geodesic integrator
3. Render black hole with accretion disk
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from geodesic_physics import GeodesicIntegrator
from raytracing_engine import BlackHoleRenderer, Camera

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='General Relativity Black Hole Raytracer')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=600, help='Image height')
    parser.add_argument('--mass', type=float, default=1.0, help='Black hole mass')
    parser.add_argument('--distance', type=float, default=20.0, help='Observer distance')
    parser.add_argument('--preview', action='store_true', help='Render low-res preview only')
    parser.add_argument('--output', type=str, default='black_hole', help='Output filename prefix')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("General Relativity Black Hole Raytracer")
    print("=" * 60)
    print(f"Black hole mass: {args.mass} M")
    print(f"Observer distance: {args.distance} M")
    print(f"Image resolution: {args.width}x{args.height}")
    print()
    
    # Initialize renderer
    renderer = BlackHoleRenderer(black_hole_mass=args.mass)
    # Set up camera for optimal black hole viewing
    # Position observer at a good distance and angle to see the complete scene
    observer_distance = args.distance  # Use the distance specified by the user
    
    # Position slightly above the disk plane and at an angle for best visual effect
    theta_offset = 0.3  # About 17 degrees above equatorial plane
    observer_position = np.array([0.0, observer_distance, np.pi/2 + theta_offset, 0.0])
    
    # Look directly at the black hole center
    look_at_point = np.array([0.0, 0.0, np.pi/2, 0.0])
    up_vector = np.array([0.0, 0.0, 1.0])  # z-direction up

    # Create camera with narrow field of view for focused black hole view
    camera = Camera(
        position=observer_position,
        look_at=look_at_point,
        up_vector=up_vector,
        fov=np.pi/8,  # 22.5 degrees field of view for focused view with less distortion
        width=args.width,
        height=args.height
    )
    
    # Generate output filename
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{args.output}.png"
    
    # Render the image
    print(f"Setting up camera at distance {observer_distance:.1f}M...")
    print(f"Camera position: r={observer_position[1]:.1f}, θ={observer_position[2]*180/np.pi:.1f}°")
    print()
    
    if args.preview:
        print("Rendering low-resolution preview...")
        image = renderer.render_quick_preview(camera)
    else:
        print("Rendering full-resolution image...")
        image = renderer.render_image(camera, output_filename)
    
    print(f"Rendering complete! Saved as {output_filename}")
    
    print("\nRendering Summary:")
    print("=" * 60)
    print(f"Main image: {output_filename}")
    
    print("\nParameters used:")
    print(f"  - Black hole mass: {args.mass} M")
    print(f"  - Observer distance: {args.distance} M")
    print(f"  - Event horizon: r = {2 * args.mass:.1f} M")
    print(f"  - Photon sphere: r = {3 * args.mass:.1f} M")
    print(f"  - ISCO: r = {6 * args.mass:.1f} M")

def quick_demo():
    """
    Quick demonstration function for immediate results.
    """
    print("Running quick black hole raytracing demo...")
    
    # Small, fast render with optimal viewing parameters
    renderer = BlackHoleRenderer(black_hole_mass=1.0)
    
    # Position camera for dramatic black hole view
    observer_distance = 12.0  # Very close for strong lensing effects
    theta_offset = 0.3  # Above the disk plane
    
    camera = Camera(
        position=np.array([0.0, observer_distance, np.pi/2 + theta_offset, 0.0]),
        look_at=np.array([0.0, 0.0, np.pi/2, 0.0]),   # Looking at black hole center
        up_vector=np.array([0.0, 0.0, 1.0]),
        fov=np.pi/4,  # 45 degrees for good scene coverage
        width=400,
        height=300
    )
    
    image = renderer.render_image(camera, "output/demo_black_hole.png")
    print("Demo complete! Check output/demo_black_hole.png")
    
    return image

if __name__ == "__main__":
    # Check if run with no arguments for quick demo
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick demo...")
        print("For full options, run: python main.py --help")
        print()
        quick_demo()
    else:
        main()
