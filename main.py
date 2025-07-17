#!/usr/bin/env python3
"""
Main script for rendering black hole images using general relativity raytracing.

This script demonstrates the complete pipeline:
1. Set up camera and scene
2. Initialize geodesic integrator
3. Render black hole with accretion disk
4. Analyze geodesic properties
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from geodesic_physics import GeodesicIntegrator
from raytracing_engine import BlackHoleRenderer, Camera
from visualization_tools import GeodesicVisualizer

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='General Relativity Black Hole Raytracer')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=600, help='Image height')
    parser.add_argument('--mass', type=float, default=1.0, help='Black hole mass')
    parser.add_argument('--distance', type=float, default=20.0, help='Observer distance')
    parser.add_argument('--preview', action='store_true', help='Render low-res preview only')
    parser.add_argument('--analyze', action='store_true', help='Generate geodesic analysis plots')
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
    
    # Set up camera
    # Position observer at distance 'distance' along positive z-axis
    observer_position = np.array([0.0, args.distance, np.pi/2, 0.0])  # [t, r, theta, phi]
    look_at_point = np.array([0.0, 2.0, np.pi/2, 0.0])  # Look toward black hole
    up_vector = np.array([0.0, 0.0, 1.0])  # z-direction up
    
    camera = Camera(
        position=observer_position,
        look_at=look_at_point,
        up_vector=up_vector,
        fov=np.pi/3,  # 60 degrees field of view
        width=args.width,
        height=args.height
    )
    
    print(f"Camera positioned at r = {args.distance:.1f} M")
    print(f"Schwarzschild radius: {2 * args.mass:.1f} M")
    print(f"Photon sphere radius: {3 * args.mass:.1f} M")
    print()
    
    # Render image
    if args.preview:
        print("Rendering low-resolution preview...")
        image = renderer.render_quick_preview(camera)
        output_filename = f"{args.output}_preview.png"
    else:
        print("Rendering full-resolution image...")
        output_filename = f"{args.output}.png"
        image = renderer.render_image(camera, output_filename)
    
    print(f"Rendering complete! Saved as {output_filename}")
    
    # Optional: Generate analysis plots
    if args.analyze:
        print("\nGenerating geodesic analysis plots...")
        visualizer = GeodesicVisualizer(black_hole_mass=args.mass)
        
        # 1. Photon sphere analysis
        r_photon, photon_fig = visualizer.analyze_photon_sphere()
        photon_fig.savefig(f"{args.output}_photon_sphere.png", dpi=150, bbox_inches='tight')
        plt.close(photon_fig)
        print(f"Photon sphere analysis saved as {args.output}_photon_sphere.png")
        
        # 2. Sample geodesic paths
        print("Computing sample geodesic trajectories...")
        
        # Different types of geodesics
        geodesic_configs = [
            {
                'name': 'Direct Impact',
                'initial_r': 50.0,
                'impact_param': 0.5 * args.mass,
                'description': 'Ray heading directly toward black hole'
            },
            {
                'name': 'Grazing Ray',
                'initial_r': 50.0,
                'impact_param': 3.5 * args.mass,
                'description': 'Ray grazing near photon sphere'
            },
            {
                'name': 'Distant Flyby',
                'initial_r': 50.0,
                'impact_param': 10.0 * args.mass,
                'description': 'Ray with large impact parameter'
            }
        ]
        
        for config in geodesic_configs:
            # Set up initial conditions
            r_init = config['initial_r'] * args.mass
            b = config['impact_param']
            
            initial_conditions = np.array([
                0.0, r_init, np.pi/2, 0.0,  # [t, r, theta, phi]
                1.0, -1.0, 0.0, b/(r_init*r_init)  # [dt, dr, dtheta, dphi]
            ])
            
            # Plot geodesic
            geodesic_fig = visualizer.plot_geodesic_3d(
                initial_conditions, 
                title=f"Geodesic: {config['name']}"
            )
            
            filename = f"{args.output}_geodesic_{config['name'].lower().replace(' ', '_')}.png"
            geodesic_fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(geodesic_fig)
            print(f"Geodesic plot saved as {filename}")
        
        # 3. Light deflection analysis
        print("Analyzing light ray deflection...")
        impact_params = np.logspace(np.log10(2*args.mass), np.log10(50*args.mass), 20)
        deflection_fig = visualizer.plot_light_ray_deflection(impact_params)
        deflection_fig.savefig(f"{args.output}_deflection.png", dpi=150, bbox_inches='tight')
        plt.close(deflection_fig)
        print(f"Deflection analysis saved as {args.output}_deflection.png")
        
        print("\nAnalysis complete!")
    
    # Display summary
    print("\n" + "=" * 60)
    print("Rendering Summary:")
    print("=" * 60)
    print(f"Main image: {output_filename}")
    
    if args.analyze:
        print("Analysis plots:")
        print(f"  - Photon sphere: {args.output}_photon_sphere.png")
        print(f"  - Geodesics: {args.output}_geodesic_*.png")
        print(f"  - Light deflection: {args.output}_deflection.png")
    
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
        position=np.array([0.0, 15.0, np.pi/2, 0.0]),
        look_at=np.array([0.0, 2.0, np.pi/2, 0.0]),
        up_vector=np.array([0.0, 0.0, 1.0]),
        fov=np.pi/4,
        width=400,
        height=300
    )
    
    image = renderer.render_image(camera, "demo_black_hole.png")
    print("Demo complete! Check demo_black_hole.png")
    
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
