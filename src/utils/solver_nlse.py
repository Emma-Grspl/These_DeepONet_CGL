import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from tqdm import tqdm

class NLSESolver:
    def __init__(self, cfg, nr=2000, nz=2000):
        """
        Solveur Numérique Split-Step Finite Difference pour la NLSE :
        dE/dz = (i/2k)*Laplacian(E) + i*Kerr - i*Plasma - Abs
        """
        # --- 1. Paramètres Physiques ---
        self.k = cfg.physics['k']
        self.r_max = cfg.physics['r_max']
        self.z_max = cfg.physics['z_max']
        
        # Coefficients NLSE (Récupérés de la config)
        # On utilise .get() avec des valeurs par défaut au cas où
        coeffs = cfg.physics.get('coefficients', {})
        self.K_order = coeffs.get('K', 8)
        self.C_kerr = coeffs.get('C_kerr', 1.0186)
        self.C_plasma = coeffs.get('C_plasma', 1.0552)
        self.C_abs = coeffs.get('C_abs', 3.4925e-06)

        # --- 2. Maillage ---
        self.Nr = nr
        self.Nz = nz
        
        self.r = np.linspace(0, self.r_max, nr)
        self.dr = self.r[1] - self.r[0]

        self.z = np.linspace(0, self.z_max, nz)
        self.dz = self.z[1] - self.z[0]

        # --- 3. Matrices pour l'étape Linéaire (Crank-Nicolson) ---
        self._setup_linear_step()

    def _setup_linear_step(self):
        """
        Prépare les matrices pour l'étape de diffraction pure.
        On utilise un demi-pas (dz/2) pour le Strang Splitting.
        """
        # Alpha pour un DEMI-PAS (dz/2)
        # Equation: dE/dz = (i/2k) * Laplacian
        # Schema CN: (I - M) E_new = (I + M) E_old
        # Facteur numérique devant le laplacien discret : i * (dz/2) / (2k * dr^2)
        
        dz_step = self.dz / 2.0 
        alpha = (1j * dz_step) / (2.0 * self.k * self.dr**2)

        # Indices pour le Laplacien Cylindrique (1/r * d/dr)
        # On évite r=0 ici, traité spécifiquement
        j = np.arange(1, self.Nr - 1)
        
        # Coefficients tridiagonaux
        lower_diag = alpha * (1.0 - 0.5 / j)
        main_diag  = alpha * (-2.0) * np.ones(self.Nr)
        upper_diag = alpha * (1.0 + 0.5 / j)

        # Assemblage Matrice M
        data = np.zeros((3, self.Nr), dtype=complex)
        data[2, 1:-1] = upper_diag 
        data[1, 1:-1] = main_diag[1:-1]
        data[0, 2:]   = lower_diag

        # Conditions aux Limites (BC)
        # r=0 (Neumann : symétrie axiale -> dE/dr = 0)
        # Approximation : E[-1] = E[1], donc le terme en (E[1]-2E[0]+E[-1]) devient 2(E[1]-E[0])
        # Le terme 1/r dE/dr tend vers d²E/dr² (L'Hôpital), donc le Laplacien vaut 2*d²E/dr²
        data[1, 0] = -4.0 * alpha  # -2 * 2 * alpha
        data[2, 1] =  4.0 * alpha  #  2 * 2 * alpha
        
        # r=r_max (Dirichlet : E=0)
        data[1, -1] = 0.0 # On force le bord à rester figé (ou 0 par défaut dans le solveur)

        M = sparse.spdiags(data, [-1, 0, 1], self.Nr, self.Nr)
        I = sparse.eye(self.Nr, format='csc')
        
        # Matrices Crank-Nicolson : (I - M/2) * E_new = (I + M/2) * E_old
        # Note : Le facteur 0.5 du schéma CN est standard.
        self.Mat_Left  = I - 0.5 * M
        self.Mat_Right = I + 0.5 * M

        # Factorisation LU pour inversion rapide (on ne le fait qu'une fois)
        self.LU = splu(self.Mat_Left.tocsc())

    def _nonlinear_step(self, E_field):
        """
        Applique l'opérateur non-linéaire sur un pas complet dz.
        Solution exacte locale : E(z+dz) = E(z) * exp(Operateur_NL * dz)
        """
        # Intensité I = |E|^2
        I = np.abs(E_field)**2
        
        # 1. Kerr (Auto-focalisation) : + i * C_kerr * I
        op_kerr = 1j * self.C_kerr * I
        
        # 2. Plasma (Défocalisation) : - i * C_plasma * I^K
        # K=8 : attention aux valeurs > 1, ça monte très vite
        I_K = np.power(I, self.K_order)
        op_plasma = -1j * self.C_plasma * I_K
        
        # 3. Absorption (Perte d'énergie) : - C_abs * I^(K-1)
        I_Km1 = np.power(I, self.K_order - 1)
        op_abs = -self.C_abs * I_Km1
        
        # Opérateur Total (D_NL)
        D_NL = op_kerr + op_plasma + op_abs
        
        # Application de l'opérateur (multiplication exponentielle)
        return E_field * np.exp(D_NL * self.dz)

    def solve(self, A_val, w0_val, f_val):
        """
        Exécute la propagation complète (Split-Step).
        Retourne : z (1D), r (1D), E (2D: Nz x Nr)
        """
        E = np.zeros((self.Nz, self.Nr), dtype=complex)

        # --- Condition Initiale (z=0) ---
        # Faisceau Gaussien focalisé
        arg_gauss = -(self.r**2) / (w0_val**2)
        phase = -(self.k * self.r**2) / (2.0 * f_val)
        E[0, :] = np.sqrt(A_val) * np.exp(arg_gauss) * np.exp(1j * phase)

        u = E[0, :].copy()

        # --- Boucle de Propagation ---
        # On utilise tqdm pour voir la barre de progression si lancé en interactif
        iterator = range(self.Nz - 1)
        # Si on veut réduire le bruit dans les logs HPC, on peut enlever tqdm
        # iterator = tqdm(iterator, desc="Solving NLSE")

        for n in iterator:
            # 1. Premier demi-pas Linéaire (Diffraction) -> dz/2
            b = self.Mat_Right.dot(u)
            u_half_1 = self.LU.solve(b)
            
            # 2. Pas complet Non-Linéaire (Physique) -> dz
            u_nl = self._nonlinear_step(u_half_1)
            
            # 3. Deuxième demi-pas Linéaire (Diffraction) -> dz/2
            b = self.Mat_Right.dot(u_nl)
            u_final = self.LU.solve(b)
            
            # Sauvegarde
            E[n+1, :] = u_final
            u = u_final

        return self.z, self.r, E