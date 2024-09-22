from tensorflow.keras.constraints import Constraint
import sympy as sp
from sympy import symbols, Function, sqrt, lambdify
# Constants
import numpy as np
from scipy.integrate import quad

# Define symbols
G, M, r, theta, phi = symbols('G M r theta phi')
class LawsOfPhysics():
    def super_init__(self, Laws=True):
        self.G = G
        self.M = M
        self.r = r
        self.theta = theta
        self.phi = phi
        self.x = x
        self.y = y
        self.z = z
        if relativity(G, M, r, theta, phi) == True:
            if conservation_of_energy(G, M, r, theta, phi) == True:
                if newtonian_laws(G, M, r, theta, phi) == True:
                    if Volumetrics(G, M, r, theta, phi) == True:
                        if electromagnetics(G, M, r, theta, phi) == True:
                             Laws = True
                        else:
                            Laws = False
    class relativity():
        def schwarzschild_metric(G, M, r, theta, phi):
          g_tt = -(1 - 2 * G * M / r)
          g_rr = 1 / (1 - 2 * G * M / r)
          g_thetatheta = r**2
          g_phiphi = r**2 * math.sin(theta)**2
          return g_tt, g_rr, g_thetatheta, g_phiphi


          # Convert the symbolic representation to a numerical function
          schwarzschild_metric_num = sp.lambdify((G, M, r, theta, phi), schwarzschild_metric(G, M, r, theta, phi))

    class ConservationOfEnergy(Constraint):
        def __call__(self, weights):
            # Assuming `weights` represents the energy state before and after an event
            energy_before = weights[0]
            energy_after = weights[1]
            return weights / (energy_after + 1e-8) * energy_before  # Ensure conservation

    class newtonianLaws():
        def newtons_second_law(force, mass):
          # Calculate acceleration (a) from force (F) and mass (m): F = ma
            acceleration = force / mass
            return acceleration

    #class maxwellEquation():

    class Volumetrics():     #
        def ideal_gas_law(pressure, volume, moles, temperature):
            # PV = nRT
            # R is the ideal gas constant
            R = 8.314  # J/(mol·K)
            # This function returns the pressure.
            pressure = (moles * R * temperature) / volume
            return pressure


    class electromagnetics():
        def gauss_law(charge_density, permittivity):
            #∇·E = ρ/ε₀
            # Assuming a symmetrical charge distribution and uniform field, return electric field.
            electric_field = charge_density / permittivity
            return electric_field

        def ampere_law(current_density, permeability):
            # ∇×B = μ₀J
            # Assuming a long straight wire with steady current, return magnetic field.
            magnetic_field = permeability * current_density
            return magnetic_field
        



# Constants
H0 = 76# Hubble constant in km/s/Mpc
H0 = H0 * (1 / 3.086e19)  # Convert H0 to 1/s

# Density parameters
Omega_m = 0.3
Omega_Lambda = 0.7

# Integrand function
def integrand(a, Omega_m, Omega_Lambda, H0):
    return 2.0 / (H0 * np.sqrt(Omega_m / a**3 + Omega_Lambda))

# Integral from a tiny value near zero to 1
epsilon = 1e-5
age_of_universe, _ = quad(integrand, epsilon, 1, args=(Omega_m, Omega_Lambda, H0))

# Convert age from seconds to gigayears
age_of_universe_gyr = age_of_universe / (3.1536e16)
print(f"Age of the Universe: {age_of_universe_gyr:.2f} billion years")
