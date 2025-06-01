import numpy as np
import matplotlib.pyplot as plt
from app import IsingModel

def run_phase_sweep(T_range, sweeps=5000, record_every=10, discard_frac=0.2, L=100):
    magnetizations = []
    energies = []
    susceptibilities = []
    specific_heats = []

    for T in T_range:
        print(f"Running T = {T:.3f}")
        model = IsingModel(L=L, T=T, init_type="all_up", seed=42)
        model.run(sweeps=sweeps, record_every=record_every, )

        discard = int(len(model.magnetization_history) * discard_frac)
        m_arr = np.array(model.magnetization_history[discard:])
        e_arr = np.array(model.energy_history[discard:])

        m_mean = np.mean(m_arr)
        e_mean = np.mean(e_arr)
        m2_mean = np.mean(m_arr ** 2)
        e2_mean = np.mean(e_arr ** 2)

        N = L * L
        beta = 1.0 / T

        chi = N * (m2_mean - m_mean**2)
        C = N * (e2_mean - e_mean**2) * beta**2

        magnetizations.append(m_mean)
        energies.append(e_mean)
        susceptibilities.append(chi)
        specific_heats.append(C)

    return {
        "T": T_range,
        "magnetization": magnetizations,
        "energy": energies,
        "susceptibility": susceptibilities,
        "specific_heat": specific_heats
    }