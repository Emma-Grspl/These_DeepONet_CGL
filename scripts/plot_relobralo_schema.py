import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def add_box(ax, xy, width, height, text, fc="#f5f7fb", ec="#1f2937", fontsize=11):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center", fontsize=fontsize)


def main():
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "relobralo_schema.png")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.2, 1.0], hspace=0.28, wspace=0.20)

    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.axis("off")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)

    add_box(ax_flow, (0.02, 0.56), 0.16, 0.22, "1. Calcul des blocs\nde loss\n$L_{pde}$ et $L_{aux}$", fc="#fde2e4", ec="#7f1d1d")
    add_box(ax_flow, (0.23, 0.56), 0.16, 0.22, "2. Mise à jour des EMA\n$\\bar L_{pde},\\bar L_{aux}$", fc="#fecdd3", ec="#7f1d1d")
    add_box(
        ax_flow,
        (0.43, 0.41),
        0.23,
        0.92,
        "3. Calcul des poids cibles\n$w^*_{pde}=\\dfrac{\\bar L_{tot}}{2\\bar L_{pde}+\\varepsilon}$\n$w^*_{aux}=\\dfrac{\\bar L_{tot}}{2\\bar L_{aux}+\\varepsilon}$\navec saturation à 5",
        fc="#fda4af",
        ec="#7f1d1d",
        fontsize=10.5,
    )
    add_box(
        ax_flow,
        (0.72, 0.49),
        0.25,
        0.36,
        "4. Lissage des poids\n$w \\leftarrow \\alpha w +(1-\\alpha) w^*$\npuis optimisation\n$L_{tot}=w_{pde}L_{pde}+w_{aux}L_{aux}$",
        fc="#fb7185",
        ec="#7f1d1d",
        fontsize=10.5,
    )

    arrowprops = dict(arrowstyle="->", lw=2.2, color="#7f1d1d")
    ax_flow.annotate("", xy=(0.23, 0.67), xytext=(0.18, 0.67), arrowprops=arrowprops)
    ax_flow.annotate("", xy=(0.44, 0.67), xytext=(0.39, 0.67), arrowprops=arrowprops)
    ax_flow.annotate("", xy=(0.72, 0.67), xytext=(0.66, 0.67), arrowprops=arrowprops)

    ax_flow.text(
        0.5,
        0.20,
        "Interprétation : le bloc dont l'EMA reste élevée reçoit un poids plus grand,\nce qui réoriente l'optimisation vers la contrainte la plus en retard.",
        ha="center",
        va="center",
        fontsize=13,
        color="#4c0519",
    )

    ax_curves = fig.add_subplot(gs[1, 0])
    steps = np.arange(1, 51)
    pde = 0.22 * np.exp(-steps / 16.0) + 0.020
    aux = 0.14 * np.exp(-steps / 5.0) + 0.006
    ema_alpha = 0.90

    pde_ema = np.zeros_like(pde)
    aux_ema = np.zeros_like(aux)
    pde_ema[0] = pde[0]
    aux_ema[0] = aux[0]
    for i in range(1, len(steps)):
        pde_ema[i] = ema_alpha * pde_ema[i - 1] + (1.0 - ema_alpha) * pde[i]
        aux_ema[i] = ema_alpha * aux_ema[i - 1] + (1.0 - ema_alpha) * aux[i]

    ax_curves.plot(steps, pde, label=r"$L_{pde}$", color="#be123c", lw=2.5)
    ax_curves.plot(steps, aux, label=r"$L_{aux}$", color="#e11d48", lw=2.5)
    ax_curves.plot(steps, pde_ema, "--", color="#881337", lw=2.0, label=r"$\bar L_{pde}$")
    ax_curves.plot(steps, aux_ema, "--", color="#9f1239", lw=2.0, label=r"$\bar L_{aux}$")
    ax_curves.set_title("Exemple : un bloc décroît plus lentement", fontsize=13)
    ax_curves.set_xlabel("Itérations")
    ax_curves.set_ylabel("Valeur de loss")
    ax_curves.grid(alpha=0.25)
    ax_curves.legend(frameon=False)

    ax_weights = fig.add_subplot(gs[1, 1])
    total = pde_ema + aux_ema + 1e-9
    w_pde_star = np.minimum(total / (2.0 * pde_ema + 1e-9), 5.0)
    w_aux_star = np.minimum(total / (2.0 * aux_ema + 1e-9), 5.0)

    w_pde = np.zeros_like(w_pde_star)
    w_aux = np.zeros_like(w_aux_star)
    w_pde[0] = 1.0
    w_aux[0] = 1.0
    for i in range(1, len(steps)):
        w_pde[i] = ema_alpha * w_pde[i - 1] + (1.0 - ema_alpha) * w_pde_star[i]
        w_aux[i] = ema_alpha * w_aux[i - 1] + (1.0 - ema_alpha) * w_aux_star[i]

    ax_weights.plot(steps, w_pde, color="#be123c", lw=2.5, label=r"$w_{pde}$")
    ax_weights.plot(steps, w_aux, color="#e11d48", lw=2.5, label=r"$w_{aux}$")
    ax_weights.axhline(1.0, color="#9ca3af", lw=1.2, ls=":")
    ax_weights.set_title("Poids adaptatifs", fontsize=13)
    ax_weights.set_xlabel("Itérations")
    ax_weights.set_ylabel("Poids")
    ax_weights.grid(alpha=0.25)
    ax_weights.legend(frameon=False)

    fig.suptitle("Équilibrage adaptatif de type RELOBRALO par EMA", fontsize=16, y=0.98)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
