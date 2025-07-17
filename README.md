# Geodesic Raytracer

A Python-based general relativity raytracer for visualizing black holes using accurate geodesic calculations.

## Overview

This project implements a physics-accurate raytracer that:
- Computes light ray geodesics in curved spacetime around black holes
- Uses the Schwarzschild metric for spherically symmetric black holes
- Renders realistic black hole images with gravitational lensing effects
- Includes accretion disk visualization
- Provides tools for analyzing geodesic properties

## Features

### Core Physics Engine (`geodesic_physics.py`)
- **Schwarzschild Metric**: Complete implementation of the Schwarzschild solution to Einstein's field equations
- **Christoffel Symbols**: Automatic calculation of connection coefficients for curved spacetime
- **Geodesic Integration**: 4th-order Runge-Kutta integration of the geodesic equation
- **Numba Optimization**: JIT-compiled functions for high-performance computation

### Raytracing Engine (`raytracing_engine.py`)
- **Camera System**: Configurable observer position and viewing parameters
- **Ray Generation**: Accurate ray generation from camera pixels
- **Accretion Disk Model**: Simple but realistic disk emission model
- **Background Stars**: Procedural star field for realistic appearance
- **Image Rendering**: Complete rendering pipeline with progress tracking

### Visualization Tools (`visualization_tools.py`)
- **3D Geodesic Plots**: Visualize light ray paths in 3D space
- **Photon Sphere Analysis**: Explore critical orbits around black holes
- **Light Deflection Studies**: Analyze gravitational lensing effects
- **Effective Potential Plots**: Understand orbital mechanics in curved spacetime

### Main Application (`main.py`)
- Command-line interface for rendering
- Configurable parameters (mass, distance, resolution)
- Batch analysis and visualization generation
- Quick demo mode for immediate results

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd geodesic-raytracer
```

2. **Create virtual environment (recommended):**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Simple Demo
Run without arguments for a quick demonstration:
```bash
python main.py
```
This generates `demo_black_hole.png` with default parameters.

### Basic Rendering
```bash
python main.py --width 800 --height 600 --mass 1.0 --distance 20.0
```

### Full Analysis
Generate detailed physics analysis along with the image:
```bash
python main.py --analyze --output my_black_hole
```

### Command Line Options
- `--width, --height`: Image resolution (default: 800x600)
- `--mass`: Black hole mass in geometric units (default: 1.0)
- `--distance`: Observer distance from black hole (default: 20.0)
- `--preview`: Generate low-resolution preview only
- `--analyze`: Create additional analysis plots
- `--output`: Output filename prefix

## Physics Background

### Schwarzschild Metric
The Schwarzschild metric describes spacetime around a spherically symmetric mass:

```
ds² = -(1-2M/r)dt² + (1-2M/r)⁻¹dr² + r²dθ² + r²sin²θdφ²
```

Where:
- `M` is the black hole mass (in geometric units where G=c=1)
- `r` is the radial coordinate
- `θ, φ` are angular coordinates

### Key Radii
- **Event Horizon**: `rs = 2M` (Schwarzschild radius)
- **Photon Sphere**: `r = 3M` (unstable circular photon orbits)
- **ISCO**: `r = 6M` (innermost stable circular orbit for massive particles)

### Geodesic Equation
Light rays follow null geodesics governed by:

```
d²xᵘ/dλ² + Γᵘᵥₚ (dxᵥ/dλ)(dxᵖ/dλ) = 0
```

Where `Γᵘᵥₚ` are the Christoffel symbols encoding spacetime curvature.

## Code Structure

```
geodesic-raytracer/
├── geodesic_physics.py     # Core GR physics implementation
├── raytracing_engine.py    # Rendering and camera systems
├── visualization_tools.py  # Analysis and plotting utilities
├── main.py                 # Main application interface
├── test_raytracer.py      # Unit tests and validation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Examples

### Basic Black Hole Rendering
```python
from geodesic_physics import GeodesicIntegrator
from raytracing_engine import BlackHoleRenderer, Camera
import numpy as np

# Create renderer
renderer = BlackHoleRenderer(black_hole_mass=1.0)

# Set up camera
camera = Camera(
    position=np.array([0.0, 15.0, np.pi/2, 0.0]),  # [t, r, θ, φ]
    look_at=np.array([0.0, 2.0, np.pi/2, 0.0]),
    up_vector=np.array([0.0, 0.0, 1.0]),
    fov=np.pi/3,  # 60 degrees
    width=800,
    height=600
)

# Render image
image = renderer.render_image(camera, "black_hole.png")
```

### Geodesic Analysis
```python
from visualization_tools import GeodesicVisualizer
import numpy as np

# Create visualizer
viz = GeodesicVisualizer(black_hole_mass=1.0)

# Analyze photon sphere
r_photon, fig = viz.analyze_photon_sphere()
fig.savefig("photon_sphere.png")

# Plot sample geodesic
initial_conditions = np.array([
    0.0, 20.0, np.pi/2, 0.0,  # [t, r, θ, φ]
    1.0, -1.0, 0.0, 0.05      # [dt, dr, dθ, dφ]
])

geodesic_fig = viz.plot_geodesic_3d(initial_conditions)
geodesic_fig.savefig("geodesic.png")
```

## Testing

Run the test suite to verify physics implementation:
```bash
python test_raytracer.py
```

This includes:
- Unit tests for all major components
- Physics validation (conservation laws, analytical comparisons)
- Integration accuracy checks

## Performance Notes

- The code uses Numba JIT compilation for performance-critical physics calculations
- Typical render times: ~1-10 minutes for 800x600 images (depending on CPU)
- Memory usage scales with image resolution
- For faster preview renders, use `--preview` flag

## Scientific Accuracy

This implementation includes:
- ✅ Exact Schwarzschild metric
- ✅ Proper geodesic integration
- ✅ Gravitational lensing effects
- ✅ Light ray bending
- ✅ Event horizon visualization
- ✅ Photon sphere effects

Limitations:
- Schwarzschild (non-rotating) black holes only
- No Doppler beaming effects
- Simplified accretion disk model
- No relativistic plasma effects

## Educational Value

This project demonstrates:
- General relativity in action
- Numerical integration of differential equations
- Ray tracing techniques
- Scientific visualization
- High-performance Python with Numba

## Future Enhancements

Potential improvements:
- Kerr metric (rotating black holes)
- More sophisticated accretion disk physics
- Doppler shifting and relativistic beaming
- Adaptive integration timesteps
- GPU acceleration
- Interactive visualization

## References

1. Misner, Thorne, Wheeler: "Gravitation" (1973)
2. Chandrasekhar: "The Mathematical Theory of Black Holes" (1983)
3. James et al.: "Gravitational lensing by spinning black holes" (2015)
4. Event Horizon Telescope Collaboration papers (2019-2022)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

**Note**: This is a educational/research implementation. For production visualization work, consider more specialized tools like GYOTO or RAyY.
