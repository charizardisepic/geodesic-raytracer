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
    # Position observer closer for better black hole visibility
    observer_distance = max(8.0, args.distance * 0.4)  # Closer viewing for better visibility
    observer_position = np.array([0.0, observer_distance, np.pi/2, 0.0])  # [t, r, theta, phi]
    look_at_point = np.array([0.0, 2.5, np.pi/2, 0.0])  # Look toward black hole center
    up_vector = np.array([0.0, 0.0, 1.0])  # z-direction up
    
    # Create camera
    camera = Camera(
        position=observer_position,
        look_at=look_at_point,
        up_vector=up_vector,
        fov=np.pi/3,  # 60 degrees field of view
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
    
    # Small, fast render
    renderer = BlackHoleRenderer(black_hole_mass=1.0)
    
    camera = Camera(
        position=np.array([0.0, 10.0, np.pi/2, 0.0]),  # Observer at r=10
        look_at=np.array([0.0, 0.0, np.pi/2, 0.0]),   # Looking at black hole
        up_vector=np.array([0.0, 0.0, 1.0]),
        fov=np.pi/3,  # 60 degrees
        width=400,
        height=300
    )
    
    image = renderer.render_image(camera, "output/output.png")
    print("Demo complete! Check output/output.png")
    
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
