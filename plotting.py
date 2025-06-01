import matplotlib.pyplot as plt
import numpy as np

def plot_observables(models, observable="energy", labels=None, title=None):
    """
    Plot energy or magnetization history from multiple IsingModel instances.
    """
    if observable not in ["energy", "magnetization"]:
        raise ValueError("Observable must be 'energy' or 'magnetization'")

    plt.figure(figsize=(10, 6))

    for idx, model in enumerate(models):
        data = (
            model.energy_history if observable == "energy"
            else model.magnetization_history
        )
        label = labels[idx] if labels else f"Model {idx + 1}"
        plt.plot(data, label=label)

    ylabel = "Energy per spin" if observable == "energy" else "Magnetization per spin"
    plt.xlabel("Sweep Count")
    plt.ylabel(ylabel)
    plt.title(title or f"{ylabel} vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_snapshot_grid(model, steps, title=None, save_path=None, cmap="RdBu"):
    """
    Plot a grid of spin lattice snapshots (3x3 or less).

    """
    n = len(steps)
    nrows = (n + 2) // 3  # ceil(n / 3)
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, step in zip(axes, steps):
        if step not in model.snapshots:
            print(f"⚠️  Warning: Step {step} not in model.snapshots.")
            ax.axis('off')
            ax.set_title(f"Missing: {step}")
            continue
        ax.imshow(model.snapshots[step], cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f"Sweep {step}")
        ax.axis('off')

    for i in range(len(steps), len(axes)):
        axes[i].axis('off')  # hide unused axes

    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt

def plot_phase_transition(results, save_prefix=None, show_exact=True):
    T = np.array(results["T"])

    # Plot magnetization
    plt.figure(figsize=(6, 4))
    plt.plot(T, results["magnetization"], 'o', label="MCMC ⟨m⟩", markersize=5)

    if show_exact:
        beta = 1 / T
        m_exact = np.zeros_like(T)
        mask = T < 2.269
        m_exact[mask] = (1 - 1 / np.sinh(2 * beta[mask]) ** 4) ** (1 / 8)
        # smooth extension: just plot m=0 after Tc
        plt.plot(T, m_exact, '-', label="Exact ⟨m⟩ (Yang)", color="black")

    plt.xlabel("Temperature T")
    plt.ylabel("Magnetization per spin")
    plt.title("Magnetization vs Temperature")
    plt.grid(True)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_magnetization.png")
    plt.show()

    # Plot energy
    plt.figure(figsize=(6, 4))
    plt.plot(T, results["energy"], 'o', label="MCMC ⟨e⟩", markersize=5, color='orange')


    plt.xlabel("Temperature T")
    plt.ylabel("Energy per spin")
    plt.title("Energy vs Temperature")
    plt.grid(True)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_energy.png")
    plt.show()

    # Susceptibility
    plt.figure(figsize=(6, 4))
    plt.plot(T, results["susceptibility"], 'o', label="χ", color="purple", markersize=5)
    plt.xlabel("Temperature T")
    plt.ylabel("Susceptibility χ")
    plt.title("Magnetic Susceptibility vs Temperature")
    plt.grid(True)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_susceptibility.png")
    plt.show()

    # Specific Heat
    plt.figure(figsize=(6, 4))
    plt.plot(T, results["specific_heat"], 'o', label="C", color="green", markersize=5)
    plt.xlabel("Temperature T")
    plt.ylabel("Specific Heat C")
    plt.title("Specific Heat vs Temperature")
    plt.grid(True)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_specific_heat.png")
    plt.show()
