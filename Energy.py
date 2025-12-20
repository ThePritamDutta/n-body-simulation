import numpy as np


def total_energy(y_history, masses, N):

    G = 6.67430e-11
    total_energies = []

    for state in y_history:
        K = 0.0
        U = 0.0

        reshaped_state = state.reshape((N, 6))

        velocities = reshaped_state[:, 3:6]

        v_sq = np.sum(velocities**2, axis=1)
        K = 0.5 * np.sum(masses * v_sq)

        positions = reshaped_state[:, 0:3]
        for i in range(N):
            # START FROM i + 1 to avoid double counting
            for j in range(i + 1, N):
                r_vec = positions[j] - positions[i]
                dist = np.linalg.norm(r_vec) + 1e4

                # U = -G * m1 * m2 / r
                U -= (G * masses[i] * masses[j]) / dist

        total_energies.append(K + U)

    return np.array(total_energies)
