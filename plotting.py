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
    plt.xlabel("Sweep index")
    plt.ylabel(ylabel)
    plt.title(title or f"{ylabel} vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()