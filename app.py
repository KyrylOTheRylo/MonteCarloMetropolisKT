import numpy as np
from numba import njit
import os
from plotting import plot_observables


@njit
def get_neighbor_sum(spins, i, j, L):
    return (
            spins[(i + 1) % L, j] +
            spins[(i - 1) % L, j] +
            spins[i, (j + 1) % L] +
            spins[i, (j - 1) % L]
    )


@njit
def metropolis_sweep(spins, beta, J, L):
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        nb_sum = get_neighbor_sum(spins, i, j, L)
        delta_E = 2 * J * s * nb_sum
        if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
            spins[i, j] *= -1


@njit
def calc_energy(spins, J, L):
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            E -= J * s * (spins[(i + 1) % L, j] + spins[i, (j + 1) % L])
    return E / (L * L)


@njit
def calc_magnetization(spins, L):
    return np.sum(spins) / (L * L)


class IsingModel:
    def __init__(self, L=100, T=2.5, J=1.0, seed=None, init_type="random"):
        self.L = L
        self.T = T
        self.beta = 1.0 / T
        self.J = J
        self.snapshots = {}
        if seed is not None:
            np.random.seed(seed)

        if init_type == "all_up":
            self.spins = np.ones((L, L), dtype=np.int8)
        elif init_type == "all_down":
            self.spins = -np.ones((L, L), dtype=np.int8)
        elif init_type == "random":
            self.spins = np.random.choice([-1, 1], size=(L, L))
        else:
            raise ValueError("init_type must be 'random', 'all_up', or 'all_down'")
        self.energy_history = []
        self.magnetization_history = []

    def run(self, sweeps=1000, record_every=10, snapshot_steps=None):
        for sweep in range(sweeps):
            metropolis_sweep(self.spins, self.beta, self.J, self.L)
            if sweep % record_every == 0:
                E = calc_energy(self.spins, self.J, self.L)
                M = calc_magnetization(self.spins, self.L)
                self.energy_history.append(E)
                self.magnetization_history.append(M)
            if snapshot_steps and sweep in snapshot_steps:
                self.snapshots[sweep] = self.spins.copy()


if __name__ == "__main__":
    model1 = IsingModel(L=100, T=2.5, J=1.0, seed=42, init_type="all_up")
    model1.run(sweeps=15000, record_every=1)
    model2 = IsingModel(L=100, T=2.5, J=1.0, seed=42, init_type="random")
    model2.run(sweeps=15000, record_every=1)
    model3 = IsingModel(L=100, T=2.5, J=1.0, seed=42, init_type="all_down")
    model3.run(sweeps=15000, record_every=1)
    plot_observables([model1, model2, model3], observable="magnetization",
                     labels=["All Up", "Random", "All Down"],
                     title="Magnetization vs Time (T=2.5, beta = 0.4)")
    plot_observables([model1, model2, model3], observable="energy",
                     labels=["All Up", "Random", "All Down"],
                     title="Magnetization vs Time (T=2.5, beta = 0.4)")
    model4 = IsingModel(L=100, T=2, J=1.0, seed=42, init_type="all_up")
    model4.run(sweeps=15000, record_every=1)
    model5 = IsingModel(L=100, T=2, J=1.0, seed=42, init_type="random")
    model5.run(sweeps=15000, record_every=1)
    model6 = IsingModel(L=100, T=2, J=1.0, seed=42, init_type="all_down")
    model6.run(sweeps=15000, record_every=1)
    model7 = IsingModel(L=100, T=2, J=1.0, seed=42, init_type="random")
    model7.run(sweeps=15000, record_every=1)
    plot_observables([model4, model5, model6, model7], observable="magnetization",
                     labels=["All Up", "Random", "All Down" , "Random 2"],
                     title="Magnetization vs Time (T=2.0, beta = 0.5)")
    plot_observables([model4, model5, model6, model7], observable="energy",
                     labels=["All Up", "Random", "All Down", "Random 2"],
                     title="Magnetization vs Time (T=2.0, beta = 0.5)")