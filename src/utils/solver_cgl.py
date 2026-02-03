import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from config import Config

class CGLSolver:
    def __init__(self, cfg, nx=512, nt=1000, bc_type='periodic'):
        """
        Solveur CGL 1D avec choix de Conditions aux Limites (BC).
        bc_type : 'periodic' (pour Chaos/Solitons) ou 'dirichlet' (pour Fronts/Tanh).
        """
        # --- 1. Paramètres Physiques ---
        self.alpha = cfg.physics['alpha']
        self.beta  = cfg.physics['beta']
        self.mu    = cfg.physics['mu']
        
        # Domaine
        x_min, x_max = cfg.physics['x_domain']
        self.L = x_max - x_min
        self.t_max = cfg.physics['t_max']

        # --- 2. Maillage ---
        self.Nx = nx
        self.Nt = nt
        
        # Grille Spatiale
        # Si Périodique : on exclut le dernier point (car x[N] = x[0])
        # Si Dirichlet : on garde tout pour fixer les bords
        if bc_type == 'periodic':
            self.x = np.linspace(x_min, x_max, nx, endpoint=False)
        else:
            self.x = np.linspace(x_min, x_max, nx, endpoint=True)
            
        self.dx = self.x[1] - self.x[0]

        self.t = np.linspace(0, self.t_max, nt)
        self.dt = self.t[1] - self.t[0]
        
        self.bc_type = bc_type

        # --- 3. Matrices pour l'étape Linéaire ---
        if self.bc_type == 'periodic':
            self._setup_linear_step_periodic()
        elif self.bc_type == 'dirichlet':
            self._setup_linear_step_dirichlet()
        else:
            raise ValueError(f"BC Type inconnu : {bc_type}")

    def _setup_linear_step_periodic(self):
        """ Crank-Nicolson Périodique """
        coeff = (1 + 1j * self.alpha) * self.dt / (2.0 * self.dx**2)
        
        main_diag = -2.0 * np.ones(self.Nx)
        off_diag  =  1.0 * np.ones(self.Nx - 1)
        
        # Laplacien avec coins périodiques
        M = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(self.Nx, self.Nx), format='lil')
        M[0, -1] = 1.0
        M[-1, 0] = 1.0
        M = M.tocsc()
        
        I = sparse.eye(self.Nx, format='csc')
        self.Mat_Left  = I - coeff * M
        self.Mat_Right = I + coeff * M
        self.LU = splu(self.Mat_Left)

    def _setup_linear_step_dirichlet(self):
        """ Crank-Nicolson Dirichlet (Bords Fixes) """
        coeff = (1 + 1j * self.alpha) * self.dt / (2.0 * self.dx**2)
        
        main_diag = -2.0 * np.ones(self.Nx)
        off_diag  =  1.0 * np.ones(self.Nx - 1)
        
        M = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(self.Nx, self.Nx), format='lil')
        
        # On force l'équation "u_new = u_old" sur les bords (indices 0 et -1)
        # En mettant les lignes à 0 ici, et en ajustant le solveur implicitement
        # (Dans la réalité CN, on applique Dirichlet via le RHS, mais ici u=constante marche ainsi)
        M[0, :] = 0.0
        M[-1, :] = 0.0
        M = M.tocsc()
        
        I = sparse.eye(self.Nx, format='csc')
        self.Mat_Left  = I - coeff * M
        self.Mat_Right = I + coeff * M
        self.LU = splu(self.Mat_Left)

    def _nonlinear_step(self, u):
        """ RK4 local pour la réaction """
        dt = self.dt
        def reaction(val):
            return self.mu * val - (1 + 1j * self.beta) * (np.abs(val)**2) * val
        
        k1 = reaction(u)
        k2 = reaction(u + 0.5 * dt * k1)
        k3 = reaction(u + 0.5 * dt * k2)
        k4 = reaction(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def solve(self, u0):
        U_history = np.zeros((self.Nt, self.Nx), dtype=complex)
        
        # Interpolation si u0 n'est pas sur la même grille
        if len(u0) != self.Nx:
            u0 = np.interp(self.x, np.linspace(self.x[0], self.x[-1], len(u0)), u0)
            
        U_history[0, :] = u0
        u = u0.astype(complex)

        for n in range(self.Nt - 1):
            # 1. Diffusion
            rhs = self.Mat_Right.dot(u)
            u_diff = self.LU.solve(rhs)
            
            # 2. Réaction
            u_next = self._nonlinear_step(u_diff)
            
            # Sécurité Dirichlet : On remet les bords à la valeur initiale (si c'est censé être fixe)
            if self.bc_type == 'dirichlet':
                u_next[0] = u0[0]
                u_next[-1] = u0[-1]
            
            u = u_next
            U_history[n+1, :] = u
            
        return self.x, U_history, self.t


def get_ground_truth_CGL(params_dict, x_min=None, x_max=None, T_max=None, Nx=None, Nt=None):
    """ Wrapper intelligent qui choisit le bon BC selon le type d'IC """
    if x_min is None: x_min = Config.x_min
    if x_max is None: x_max = Config.x_max
    if T_max is None: T_max = Config.T_max
    if Nx is None: Nx = 512
    if Nt is None: Nt = int(np.ceil(T_max / 0.005))

    # --- DÉTECTION DU TYPE DE BC ---
    # Type 2 = Tanh -> Front -> Dirichlet
    # Type 0, 1 = Gauss/Sech -> Localisé -> Périodique
    ic_type = int(params_dict.get('type', 0))
    selected_bc = 'dirichlet' if ic_type == 2 else 'periodic'

    # --- Génération IC ---
    A  = params_dict.get('A', 1.0)
    w0 = params_dict.get('w0', 1.0)
    x0 = params_dict.get('x0', 0.0)
    k  = params_dict.get('k', 1.0)

    # Note: On génère la grille selon le BC choisi (via le Solver ou manuellement ici)
    if selected_bc == 'periodic':
        x_grid = np.linspace(x_min, x_max, Nx, endpoint=False)
    else:
        x_grid = np.linspace(x_min, x_max, Nx, endpoint=True)

    X = (x_grid - x0) / (w0 + 1e-9)
    Phase = np.exp(1j * k * x_grid)
    
    if ic_type == 0:   Env = A * np.exp(-X**2)
    elif ic_type == 1: Env = A / np.cosh(X)
    elif ic_type == 2: Env = A * np.tanh(X)
    else:              Env = A * np.exp(-X**2)

    u0 = Env * Phase

    # --- Lancement du Solveur ---
    solver = CGLSolver(Config, nx=Nx, nt=Nt, bc_type=selected_bc)
    x, U_matrix, t = solver.solve(u0)

    if U_matrix.shape == (Nt, Nx):
        U_matrix = U_matrix.T

    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
    return X_grid, T_grid, U_matrix