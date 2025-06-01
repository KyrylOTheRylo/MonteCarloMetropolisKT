import matplotlib.pyplot as plt


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