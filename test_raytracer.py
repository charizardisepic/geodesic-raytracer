"""
Test suite for the geodesic raytracer components.
"""

import numpy as np
import unittest
from geodesic_physics import GeodesicIntegrator, schwarzschild_metric_components, christoffel_symbols
from raytracing_engine import BlackHoleRenderer, Camera, AccretionDisk
from visualization_tools import GeodesicVisualizer

class TestGeodesicPhysics(unittest.TestCase):
    """Test the geodesic physics implementation."""
    
    def setUp(self):
        self.M = 1.0
        self.integrator = GeodesicIntegrator(M=self.M)
    
    def test_schwarzschild_metric(self):
        """Test Schwarzschild metric components."""
        r = 10.0
        theta = np.pi/2
        
        g_tt, g_rr, g_theta_theta, g_phi_phi = schwarzschild_metric_components(r, theta, self.M)
        
        # Check expected values
        expected_g_tt = -(1 - 2*self.M/r)
        expected_g_rr = 1/(1 - 2*self.M/r)
        expected_g_theta_theta = r*r
        expected_g_phi_phi = r*r * np.sin(theta)**2
        
        self.assertAlmostEqual(g_tt, expected_g_tt, places=10)
        self.assertAlmostEqual(g_rr, expected_g_rr, places=10)
        self.assertAlmostEqual(g_theta_theta, expected_g_theta_theta, places=10)
        self.assertAlmostEqual(g_phi_phi, expected_g_phi_phi, places=10)
    
    def test_christoffel_symbols(self):
        """Test that Christoffel symbols are computed correctly."""
        r = 10.0
        theta = np.pi/2
        
        gamma = christoffel_symbols(r, theta, self.M)
        
        # Check array shape
        self.assertEqual(gamma.shape, (4, 4, 4))
        
        # Check some specific known values
        # Γᵗᵗʳ should be rs/(2r²(1-rs/r))
        rs = 2 * self.M
        expected_gamma_001 = rs / (2 * r*r * (1 - rs/r))
        self.assertAlmostEqual(gamma[0, 0, 1], expected_gamma_001, places=8)
    
    def test_geodesic_integration(self):
        """Test that geodesic integration produces reasonable results."""
        # Set up initial conditions for a light ray
        r_initial = 20.0
        initial_conditions = np.array([
            0.0, r_initial, np.pi/2, 0.0,  # [t, r, theta, phi]
            1.0, -0.5, 0.0, 0.1/r_initial  # [dt, dr, dtheta, dphi]
        ])
        
        trajectory, escaped = self.integrator.integrate_geodesic(
            initial_conditions, max_steps=1000
        )
        
        # Check that we get a reasonable trajectory
        self.assertGreater(len(trajectory), 10)
        self.assertIsInstance(escaped, bool)
        
        # Check that r values are positive
        r_values = trajectory[:, 1]
        self.assertTrue(np.all(r_values > 0))

class TestRaytracingEngine(unittest.TestCase):
    """Test the raytracing engine components."""
    
    def setUp(self):
        self.renderer = BlackHoleRenderer(black_hole_mass=1.0)
    
    def test_camera_creation(self):
        """Test camera initialization."""
        camera = Camera(
            position=np.array([0.0, 10.0, np.pi/2, 0.0]),
            look_at=np.array([0.0, 2.0, np.pi/2, 0.0]),
            up_vector=np.array([0.0, 0.0, 1.0]),
            fov=np.pi/4,
            width=100,
            height=100
        )
        
        self.assertEqual(camera.width, 100)
        self.assertEqual(camera.height, 100)
        self.assertAlmostEqual(camera.fov, np.pi/4)
    
    def test_ray_generation(self):
        """Test ray generation from camera."""
        camera = Camera(
            position=np.array([0.0, 10.0, np.pi/2, 0.0]),
            look_at=np.array([0.0, 2.0, np.pi/2, 0.0]),
            up_vector=np.array([0.0, 0.0, 1.0]),
            fov=np.pi/4,
            width=100,
            height=100
        )
        
        ray = camera.generate_ray(50, 50)  # Center pixel
        
        # Check ray structure
        self.assertEqual(len(ray.origin), 4)
        self.assertEqual(len(ray.direction), 4)
        
        # Ray origin should match camera position
        np.testing.assert_array_almost_equal(ray.origin, camera.position)
    
    def test_accretion_disk(self):
        """Test accretion disk model."""
        disk = AccretionDisk(inner_radius=3.0, outer_radius=10.0)
        
        # Test point inside disk
        r_test = 5.0
        theta_test = np.pi/2  # Equatorial plane
        phi_test = 0.0
        
        self.assertTrue(disk.is_in_disk(r_test, theta_test))
        
        color = disk.get_emission(r_test, theta_test, phi_test)
        self.assertEqual(len(color), 3)  # RGB
        self.assertTrue(all(0 <= c <= 1 for c in color))
        
        # Test point outside disk
        r_outside = 15.0
        self.assertFalse(disk.is_in_disk(r_outside, theta_test))

class TestVisualizationTools(unittest.TestCase):
    """Test visualization utilities."""
    
    def setUp(self):
        self.visualizer = GeodesicVisualizer(black_hole_mass=1.0)
    
    def test_photon_sphere_calculation(self):
        """Test photon sphere analysis."""
        r_photon, fig = self.visualizer.analyze_photon_sphere()
        
        # Photon sphere should be at 1.5 * rs = 3M
        expected_r_photon = 3.0 * self.visualizer.M
        self.assertAlmostEqual(r_photon, expected_r_photon, places=8)
        
        # Figure should be created
        self.assertIsNotNone(fig)

def run_physics_validation():
    """
    Run validation tests for the physics implementation.
    """
    print("Running physics validation tests...")
    
    # Test 1: Conservation laws
    print("\n1. Testing conservation laws...")
    integrator = GeodesicIntegrator(M=1.0)
    
    # Light ray with specific angular momentum
    r_initial = 20.0
    L = 5.0  # Angular momentum
    E = 1.0  # Energy
    
    initial_conditions = np.array([
        0.0, r_initial, np.pi/2, 0.0,
        E, -np.sqrt(E*E - (1-2/r_initial)*(L*L/(r_initial*r_initial))), 0.0, L/(r_initial*r_initial)
    ])
    
    trajectory, escaped = integrator.integrate_geodesic(initial_conditions, max_steps=2000)
    
    # Check energy conservation (approximately)
    energies = []
    angular_momenta = []
    
    for point in trajectory[::10]:  # Sample every 10th point
        t, r, theta, phi = point[:4]
        dt, dr, dtheta, dphi = point[4:]
        
        # Calculate conserved quantities
        g_tt, g_rr, g_theta_theta, g_phi_phi = schwarzschild_metric_components(r, theta, 1.0)
        
        energy = -g_tt * dt
        ang_momentum = g_phi_phi * dphi
        
        energies.append(energy)
        angular_momenta.append(ang_momentum)
    
    energy_variation = np.std(energies) / np.mean(energies)
    angular_momentum_variation = np.std(angular_momenta) / np.mean(angular_momenta)
    
    print(f"Energy conservation: {energy_variation:.2e} (relative variation)")
    print(f"Angular momentum conservation: {angular_momentum_variation:.2e} (relative variation)")
    
    if energy_variation < 1e-3 and angular_momentum_variation < 1e-3:
        print("✓ Conservation laws satisfied")
    else:
        print("✗ Conservation laws violated - check integration accuracy")
    
    # Test 2: Known analytical results
    print("\n2. Testing against analytical results...")
    
    # Light ray deflection for large impact parameter
    # Should approach 4M/b for b >> M
    b_large = 50.0  # Large impact parameter
    r_start = 100.0
    
    initial_conditions = np.array([
        0.0, r_start, np.pi/2, 0.0,
        1.0, -1.0, 0.0, b_large/(r_start*r_start)
    ])
    
    trajectory, escaped = integrator.integrate_geodesic(initial_conditions, max_steps=5000)
    
    if escaped and len(trajectory) > 100:
        initial_phi = trajectory[0][3]
        final_phi = trajectory[-1][3]
        deflection = abs(final_phi - initial_phi)
        
        analytical_deflection = 4.0 / b_large  # 4M/b with M=1
        relative_error = abs(deflection - analytical_deflection) / analytical_deflection
        
        print(f"Numerical deflection: {deflection:.6f} rad")
        print(f"Analytical deflection: {analytical_deflection:.6f} rad")
        print(f"Relative error: {relative_error:.2%}")
        
        if relative_error < 0.1:  # 10% tolerance
            print("✓ Light deflection matches analytical prediction")
        else:
            print("✗ Light deflection deviates from analytical prediction")
    
    print("\nPhysics validation complete!")

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run physics validation
    run_physics_validation()
