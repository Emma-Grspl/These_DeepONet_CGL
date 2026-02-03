import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

# On supprime l'import de 'config' qui est problématique
# from config import Config 

class CGLSolver:
    def __init__(self, cfg, params_dict=None, nx=256, nt=1000, bc_type='periodic'):
        """
        Solveur CGL 1D Généraliste (Accepte les params dynamiques).
        """
        # --- 1. Paramètres Physiques ---
        # Si params_dict est fourni (cas généraliste), on l'utilise.
        # Sinon on fallback sur la config (cas fixe).
        p = params_dict if params_dict is not None else cfg.physics
        
        self.alpha = p.get('alpha', 0.0)
        self.beta  = p.get('beta', 0.0)
        self.mu    = p.get('mu', 0.0)
        self.V     = p.get('V', 0.0) # Vitesse d'advection
        
        # Domaine
        x_min, x_max = cfg.physics['x_domain']
        self.L = x_max - x_min
        self.t_max = cfg.physics['t_max']

        # --- 2. Maillage ---
        self.Nx = nx
        self.Nt = nt
        
        if bc_type == 'periodic':
            self.x = np.linspace(x_min, x_max, nx, endpoint=False)
        else:
            self.x = np.linspace(x_min, x_max, nx, endpoint=True)
            
        self.dx = self.x[1] - self.x[0]

        self.t = np.linspace(0, self.t_max, nt)
        self.dt = self.t[1] - self.t[0]
        
        self.bc_type = bc_type

        # --- 3. Matrices pour l'étape Linéaire (Diffusion + Advection) ---
        if self.bc_type == 'periodic':
            self._setup_linear_step_periodic()
        else:
            self._setup_linear_step_dirichlet()

    def _setup_linear_step_periodic(self):
        """ Crank-Nicolson Périodique avec Advection Centrée """
        # Coefficient Diffusion
        # D = (1 + i*alpha)
        # Terme : D * d2u/dx2
        coeff_diff = (1 + 1j * self.alpha) * self.dt / (2.0 * self.dx**2)
        
        # Coefficient Advection (Nouveau !)
        # Terme : - V * du/dx
        # Schéma centré : (u_{i+1} - u_{i-1}) / 2dx
        # Coeff matrice : -V * dt / 2 * (1/2dx) = -V*dt / 4dx
        coeff_adv = -self.V * self.dt / (4.0 * self.dx)

        # Diagonales
        # Diffusion : -2 au centre, +1 sur les côtés
        # Advection : 0 au centre, +1 à droite, -1 à gauche
        
        # Diagonale Principale (Uniquement Diffusion)
        main_diag = -2.0 * coeff_diff * np.ones(self.Nx)
        
        # Diagonale Supérieure (i+1) : Diff(+1) + Adv(+1)
        upper_diag = (coeff_diff + coeff_adv) * np.ones(self.Nx - 1)
        
        # Diagonale Inférieure (i-1) : Diff(+1) + Adv(-1)
        lower_diag = (coeff_diff - coeff_adv) * np.ones(self.Nx - 1)
        
        # Construction Matrice M (termes spatiaux)
        M = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], shape=(self.Nx, self.Nx), format='lil')
        
        # Coins Périodiques
        # Coin Haut-Droit (correspond à i-1 pour i=0 -> i=N-1)
        M[0, -1] = (coeff_diff - coeff_adv)
        # Coin Bas-Gauche (correspond à i+1 pour i=N-1 -> i=0)
        M[-1, 0] = (coeff_diff + coeff_adv)
        
        M = M.tocsc()
        
        I = sparse.eye(self.Nx, format='csc')
        # CN : (I - M) u_new = (I + M) u_old
        # Attention : M contient déjà les dt/2 via les coeffs
        # J'ai mis dt/2 dans coeff_diff, donc Mat_Left = I - M est correct
        
        self.Mat_Left  = I - M
        self.Mat_Right = I + M
        self.LU = splu(self.Mat_Left)

    def _setup_linear_step_dirichlet(self):
        """ Crank-Nicolson Dirichlet (Bords Fixes) """
        coeff_diff = (1 + 1j * self.alpha) * self.dt / (2.0 * self.dx**2)
        coeff_adv = -self.V * self.dt / (4.0 * self.dx)
        
        main_diag = -2.0 * coeff_diff * np.ones(self.Nx)
        upper_diag = (coeff_diff + coeff_adv) * np.ones(self.Nx - 1)
        lower_diag = (coeff_diff - coeff_adv) * np.ones(self.Nx - 1)
        
        M = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], shape=(self.Nx, self.Nx), format='lil')
        
        # Bords Fixes (Dirichlet Homogène ou non, géré par le fix final)
        M[0, :] = 0.0
        M[-1, :] = 0.0
        M = M.tocsc()
        
        I = sparse.eye(self.Nx, format='csc')
        self.Mat_Left  = I - M
        self.Mat_Right = I + M
        self.LU = splu(self.Mat_Left)

    def _nonlinear_step(self, u):
        """ RK4 local pour la réaction """
        dt = self.dt
        # Terme source : mu*u - (1+i*beta)|u|^2 u
        def reaction(val):
            return self.mu * val - (1 + 1j * self.beta) * (np.abs(val)**2) * val
        
        k1 = reaction(u)
        k2 = reaction(u + 0.5 * dt * k1)
        k3 = reaction(u + 0.5 * dt * k2)
        k4 = reaction(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def solve(self, u0):
        U_history = np.zeros((self.Nt, self.Nx), dtype=complex)
        
        # Interpolation initiale
        if len(u0) != self.Nx:
            u0 = np.interp(self.x, np.linspace(self.x[0], self.x[-1], len(u0)), u0)
            
        U_history[0, :] = u0
        u = u0.astype(complex)

        for n in range(self.Nt - 1):
            # 1. Étape Linéaire (Diffusion + Advection)
            rhs = self.Mat_Right.dot(u)
            u_diff = self.LU.solve(rhs)
            
            # 2. Étape Non-Linéaire (Réaction)
            u_next = self._nonlinear_step(u_diff)
            
            # Sécurité Dirichlet
            if self.bc_type == 'dirichlet':
                u_next[0] = u0[0]
                u_next[-1] = u0[-1]
            
            u = u_next
            U_history[n+1, :] = u
            
        return self.x, U_history, self.t


def get_ground_truth_CGL(params_dict, x_min, x_max, T_max, Nx=256, Nt=None):
    """ 
    Wrapper compatible avec le trainer.
    Génère une classe Config temporaire factice pour satisfaire l'init du solver
    si on ne veut pas passer le gros cfg.
    """
    if Nt is None:
        # Pas de temps adaptatif pour stabilité
        # dt ~ dx^2 / 4 est prudent pour diffusion explicite, 
        # mais CN est inconditionnellement stable.
        # On prend un dt raisonnable pour la précision temporelle.
        Nt = int(np.ceil(T_max / 0.002)) 

    # On crée un objet config dummy pour passer les infos structurelles
    class DummyConfig:
        physics = {
            'x_domain': [x_min, x_max],
            't_max': T_max
        }
    cfg_dummy = DummyConfig()

    # --- DÉTECTION DU TYPE DE BC ---
    ic_type = int(params_dict.get('type', 0))
    selected_bc = 'dirichlet' if ic_type == 2 else 'periodic'

    # --- Génération IC ---
    A  = params_dict.get('A', 1.0)
    w0 = params_dict.get('w0', 1.0)
    x0 = params_dict.get('x0', 0.0)
    k  = params_dict.get('k', 1.0)

    # Grille temporaire pour l'IC
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
    # On passe params_dict pour que alpha, beta, V soient pris en compte !
    solver = CGLSolver(cfg_dummy, params_dict=params_dict, nx=Nx, nt=Nt, bc_type=selected_bc)
    
    x, U_matrix, t = solver.solve(u0)

    # Format de sortie : (Nx, Nt) -> (Nx, Nt)
    # Le solver renvoie (Nt, Nx), on transpose pour matcher la convention DeepONet souvent (X, T)
    if U_matrix.shape == (Nt, Nx):
        U_matrix = U_matrix.T

    X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
    
    return X_grid, T_grid, U_matrix