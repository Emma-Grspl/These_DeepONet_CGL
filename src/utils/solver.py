import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from tqdm import tqdm

class CrankNicolsonSolver:
    def __init__(self, cfg, nr=1000, nz=1000):
        """
        Solveur Numérique Classique pour dE/dz = (i/2k) * Laplacian(E)
        """
        # Adaptation pour lire la config YAML structurée
        self.k = cfg.physics['k']
        self.r_max = cfg.physics['r_max']
        self.z_max = cfg.physics['z_max']

        self.Nr = nr
        self.Nz = nz

        # 1. Maillage
        self.r = np.linspace(0, self.r_max, nr)
        self.dr = self.r[1] - self.r[0]

        self.z = np.linspace(0, self.z_max, nz)
        self.dz = self.z[1] - self.z[0]

        # 2. Préparation des Matrices
        self._setup_matrices()

    def _setup_matrices(self):
        # Facteur alpha pour Crank-Nicolson : alpha = i * dz / (4 * k * dr^2)
        alpha = (1j * self.dz) / (4.0 * self.k * self.dr**2)

        # --- Diagonales ---
        j = np.arange(1, self.Nr - 1)
        lower_diag = alpha * (1.0 - 0.5 / j)
        main_diag  = alpha * (-2.0) * np.ones(self.Nr)
        upper_diag = alpha * (1.0 + 0.5 / j)

        # --- Assemblage M ---
        data = np.zeros((3, self.Nr), dtype=complex)
        data[2, 1:-1] = upper_diag 
        data[1, 1:-1] = main_diag[1:-1]
        data[0, 2:]   = lower_diag

        # Conditions aux Limites (BC)
        # r=0 (Neumann)
        data[1, 0] = -4.0 * alpha
        data[2, 1] =  4.0 * alpha
        # r=r_max (Dirichlet)
        data[1, -1] = 0.0 

        M = sparse.spdiags(data, [-1, 0, 1], self.Nr, self.Nr)

        I = sparse.eye(self.Nr, format='csc')
        self.Mat_Left  = I - M
        self.Mat_Right = I + M

        # Factorisation LU
        self.LU = splu(self.Mat_Left.tocsc())

    def solve(self, A_val, w0_val, f_val):
        """
        Retourne z, r, E_field
        """
        E = np.zeros((self.Nz, self.Nr), dtype=complex)

        # Condition Initiale
        arg_gauss = -(self.r**2) / (w0_val**2)
        phase = -(self.k * self.r**2) / (2.0 * f_val)
        E[0, :] = np.sqrt(A_val) * np.exp(arg_gauss) * np.exp(1j * phase)

        u = E[0, :].copy()

        # Propagation (sans print tqdm si on veut être silencieux sur HPC, sinon laisser)
        for n in range(self.Nz - 1):
            b = self.Mat_Right.dot(u)
            u_new = self.LU.solve(b)
            E[n+1, :] = u_new
            u = u_new

        return self.z, self.r, E
