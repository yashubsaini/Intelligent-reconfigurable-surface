import numpy as np

class IRSArray:
    """
    Models the physical Reconfigurable Intelligent Surface (IRS).
    """
    def __init__(self, num_elements_x, num_elements_y, frequency=28e9):
        """
        Initialize the IRS panel.
        
        Args:
            num_elements_x (int): Number of elements in the x-direction.
            num_elements_y (int): Number of elements in the y-direction.
            frequency (float): Operating frequency in Hz (default 28 GHz).
        """
        self.Nx = num_elements_x
        self.Ny = num_elements_y
        self.N = self.Nx * self.Ny
        
        self.freq = frequency
        self.c = 3e8 # Speed of light
        self.wavelength = self.c / self.freq
        self.spacing = self.wavelength / 2.0 # Half-wavelength spacing
        
        # Current phase shifts (continuous model: 0 to 2pi)
        self.phases = np.zeros(self.N)

    def get_reflection_matrix(self):
        """
        Returns the diagonal reflection matrix Phi.
        """
        return np.diag(np.exp(1j * self.phases))

    def set_random_phases(self):
        """
        Set completely random phases (continuous).
        """
        self.phases = np.random.uniform(0, 2*np.pi, self.N)

    def optimize_phases_continuous(self, G, h_r):
        """
        Mathematically optimal continuous phase shifts to maximize cascaded channel gain.
        h_cascaded = h_r^H * Phi * G
        To maximize, we set the phase of element n such that the total phase 
        of the cascaded path through element n is aligned.
        
        Args:
            G (np.array): Channel from BS to IRS (N x M, or N x 1 for single antenna BS)
            h_r (np.array): Channel from IRS to UE (N x 1)
        """
        # Ensure 1D arrays for simplicity (assuming single antenna BS and UE for now)
        G = G.flatten()
        h_r = h_r.flatten()
        
        # We want to align the phases: phase(h_r_n^H * e^{j*theta_n} * G_n) = constant (e.g., 0)
        # So theta_n = -phase(G_n) - phase(h_r_n^H) = -phase(G_n) + phase(h_r_n)
        
        theta_opt = np.angle(h_r) - np.angle(G)
        
        # Wrap phases to [0, 2pi)
        self.phases = np.mod(theta_opt, 2*np.pi)

    def set_discrete_phases(self, num_bits):
        """
        Quantize current continuous phases to discrete levels.
        e.g., num_bits=1 -> 0, pi
        num_bits=2 -> 0, pi/2, pi, 3pi/2
        
        We will implement this fully in later phases.
        """
        levels = 2 ** num_bits
        step = (2 * np.pi) / levels
        
        # Quantize by rounding to nearest step
        quantized = np.round(self.phases / step) * step
        self.phases = np.mod(quantized, 2*np.pi)
