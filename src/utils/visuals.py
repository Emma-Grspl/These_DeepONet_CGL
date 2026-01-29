import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from types import SimpleNamespace
from src.models.Diffractive_PI_DeepOnet import PI_DeepONet_Robust

def plot_results(model_path, config_path):
    # 1. Charger Config et Modèle
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(physics=cfg_dict['physics'], model=cfg_dict['model'])
    
    model = PI_DeepONet_Robust(cfg)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. Grille d'évaluation
    z_max = cfg.physics['z_max']
    r_max = 5.0 # On zoom sur le faisceau (max est à 20)
    nz, nr = 500, 200
    
    z = torch.linspace(0, z_max, nz)
    r = torch.linspace(0, r_max, nr)
    Z, R = torch.meshgrid(z, r, indexing='ij')

    # Paramètre de test : A=1, w0=1mm, f=800mm
    params = torch.tensor([[1.0, 1.0, 800.0]]).repeat(Z.numel(), 1)
    coords = torch.stack([R.flatten(), Z.flatten()], dim=1)

    # 3. Inférence
    with torch.no_grad():
        re, im = model(params, coords)
        intensity = (re**2 + im**2).reshape(nz, nr).numpy()

    # 4. Affichage
    plt.figure(figsize=(15, 6))
    plt.imshow(intensity.T, extent=[0, z_max, 0, r_max], origin='lower', 
               aspect='auto', cmap='magma')
    plt.colorbar(label='Intensité $|A|^2$')
    plt.axvline(x=800, color='cyan', linestyle='--', label='Foyer théorique')
    
    plt.xlabel('z (mm)')
    plt.ylabel('r (mm)')
    plt.title('Propagation NLSE prédite par DeepONet')
    plt.legend()
    plt.show()
