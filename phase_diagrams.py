"""
BY CLAUDE -- WILL CLEAN UP
Phase diagram calculations using pycalphad
Provides liquidus and solidus temperature lookups for binary alloys
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
from config import PHASE_DIAGRAMS_DIR

class PhaseDiagramCalculator:
    '''
    Phase diagram calculations for binary alloys
    Computes tables for fast liquidus/solidus lookups during simulation loop
    '''

    def __init__(self, tdb_file, components, composition_points=100):
        """
        Note: components should contain 'VA'
        """

        #load database
        with open(tdb_file, encoding='iso-8859-1') as f:
            self.dbf = Database(f.read())

        self.components = components
        self.solute = [c for c in components if c != 'VA'][1] #solute = second non-VA component
        self.phases = list(self.dbf.phases.keys())

        print(f"Components: {components}")
        print(f"Solute element: {self.solute}")
        print(f"Available phases: {self.phases}")

        #lookup tables
        self.X_range = np.linspace(0.0, 1.0, composition_points)
        self.T_liquidus_table = None
        self.T_solidus_table = None

        self.build_lookup_tables()

    def build_lookup_tables(self, T_min=300, T_max=2100, T_points=50, pressure=101325):
        print(f"\nBuilding lookup tables...")

        #prepare arrays
        T_liquidus = np.zeros_like(self.X_range)
        T_solidus = np.zeros_like(self.X_range)
        temps = np.linspace(T_min, T_max, T_points)

        for i, X_solute in enumerate(self.X_range): #loop over composition values
            try:
                T_liq, T_sol = self.calculate_phase_boundaries(X_solute, temps, pressure)
                T_liquidus[i] = T_liq
                T_solidus[i] = T_sol

            except Exception as e:
                warnings.warn(f"Failed at X={X_solute:.3f}: {e}")
                # Use interpolation or default values
                if i > 0:
                    T_liquidus[i] = T_liquidus[i - 1]
                    T_solidus[i] = T_solidus[i - 1]

        self.T_liquidus_table = T_liquidus
        self.T_solidus_table = T_solidus

        print(f"Lookup tables created")

    def calculate_phase_boundaries(self, X_solute, temps, pressure, tol=0.01):
        """
        Use pycalphad to calculate phase boundaries
        tol = max fraction of liquid allowed at liquidus
        """
        #conditions
        conditions = {
            v.X(self.solute): X_solute,
            v.T: temps,
            v.P: pressure,
            v.N: 1.0
        }

        #calculate equilibrium using pycalphad
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eq = equilibrium(
                self.dbf,
                self.components,
                self.phases,
                conditions
            )

        phase_labels = eq.Phase.values.squeeze()
        phase_fractions = eq.NP.values.squeeze()  # Phase fractions

        #find T_liquidus, T_solidus
        T_liquidus = temps[-1]
        T_solidus = temps[0]

        for j in range(len(temps) - 1, -1, -1): #from high to low T
            phases = phase_labels[j]
            fractions = phase_fractions[j]

            liquid_fraction = 0.0
            for phase, frac in zip(phases, fractions):
                if 'LIQUID' in str(phase).upper():
                    liquid_fraction += frac

            #find liquidus (solid frac > tol)
            if liquid_fraction < 1-tol and T_liquidus == temps[-1]: #if solid frac condition met AND condition has not been satisfied earlier
                T_liquidus = temps[j + 1] if j < len(temps) - 1 else temps[j]

            #find solidus (liquid frac < tol)
            if liquid_fraction < tol:
                T_solidus = temps[j]
                break

        return T_liquidus, T_solidus

    def get_liquidus_solidus(self, X_solute):
        """
        Get liquidus and solidus temperatures for given composition
        Uses interpolation from pre-computed lookup tables

        Args:
            X_solute: Mole fraction of solute element (0 to 1)

        Returns:
            T_liquidus, T_solidus in Kelvin
        """
        if self.T_liquidus_table is None or self.T_solidus_table is None:
            raise RuntimeError("Lookup tables not initialized")

        # Clip to valid range
        X_solute = np.clip(X_solute, 0.0, 1.0)

        # Interpolate from lookup tables
        T_liq = np.interp(X_solute, self.X_range, self.T_liquidus_table)
        T_sol = np.interp(X_solute, self.X_range, self.T_solidus_table)

        return T_liq, T_sol

    def get_liquidus_solidus_grid(self, X_grid):
        """
        Get liquidus and solidus temperature grids for composition grid

        Args:
            X_grid: 2D array of solute mole fractions

        Returns:
            T_liquidus_grid, T_solidus_grid as 2D arrays
        """
        # Vectorized interpolation
        T_liq_grid = np.interp(X_grid.flatten(), self.X_range, self.T_liquidus_table)
        T_sol_grid = np.interp(X_grid.flatten(), self.X_range, self.T_solidus_table)

        return T_liq_grid.reshape(X_grid.shape), T_sol_grid.reshape(X_grid.shape)

    def get_partition_coefficient(self, X_solute, T=None):
        """
        Calculate partition coefficient k = X_solid / X_liquid
        at the interface for given composition

        This is a simplified approximation. For accurate k values,
        need to run equilibrium calculation at the solidus temperature.

        Args:
            X_solute: Bulk mole fraction of solute
            T: Temperature (optional, defaults to solidus)

        Returns:
            k: Partition coefficient
        """
        # Simplified: use equilibrium tie-line at solidus
        # For many systems, k is approximately constant

        # Common approximations:
        # Fe-Si: k ≈ 0.77 (Si partitions slightly to liquid)
        # Fe-C: k ≈ 0.19 (C strongly partitions to liquid)

        # Placeholder - should be calculated from phase diagram
        k = 0.77  # For Fe-Si as example

        return k

# Example usage
filename = 'Fe-Si-Zn.tdb'
filepath = PHASE_DIAGRAMS_DIR + '/' + filename

if __name__ == "__main__":
    # Example for Fe-Si system
    tdb_file = filepath  # Use your fixed TDB file
    components = ['FE', 'SI', 'VA']

    # Create calculator
    calc = PhaseDiagramCalculator(tdb_file, components, composition_points=50)

    # Query specific composition
    X_Si = 0.1  # 10 at% Si
    T_liq, T_sol = calc.get_liquidus_solidus(X_Si)
    print(f"\nFor X(Si) = {X_Si:.2f}:")
    print(f"  Liquidus: {T_liq:.1f} K")
    print(f"  Solidus: {T_sol:.1f} K")
    print(f"  Freezing range: {T_liq - T_sol:.1f} K")

    # Save tables for fast loading later
    calc.save_lookup_tables("Fe-Si_lookup_tables.npz")

    # Test with composition grid
    X_grid = np.random.uniform(0, 0.2, size=(10, 10))
    T_liq_grid, T_sol_grid = calc.get_liquidus_solidus_grid(X_grid)
    print(f"\nGrid calculations:")
    print(f"  Liquidus range: {T_liq_grid.min():.1f} - {T_liq_grid.max():.1f} K")
    print(f"  Solidus range: {T_sol_grid.min():.1f} - {T_sol_grid.max():.1f} K")