import numpy as np

def total_energy(y_history, masses, N, eps=1e4):
   

    G = 6.67430e-11
    energies = []

    for state in y_history:

        reshaped_state = state.reshape((N, 6))

        positions = reshaped_state[:, 0:3]
        velocities = reshaped_state[:, 3:6]

        # ---------- Kinetic Energy ----------
        v_sq = np.sum(velocities**2, axis=1)
        K = 0.5 * np.sum(masses * v_sq)

        # ---------- Potential Energy ----------
        U = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = positions[j] - positions[i]
                dist = np.sqrt(np.dot(r_vec, r_vec) + eps**2)
                U -= G * masses[i] * masses[j] / dist

        energies.append(K + U)

    E = np.array(energies)

    # ---------- Energy Drift ----------
    E0 = E[0]
    dE = E - E0
    dE_rel = dE / abs(E0)

    return np.array(dE_rel)

