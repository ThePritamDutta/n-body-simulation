import numpy as np


def bodies(N):
    bodies = []
    masses = []

    for i in range(N):

        # Random Position (Scale 1e9)
        px = np.random.uniform(-1.6e9, 1.6e9)
        py = np.random.uniform(-1.6e9, 1.6e9)
        pz = np.random.uniform(-1.6e9, 1.6e9)
        # Giving Random velocities (Scale 500)
        vx = np.random.uniform(-500, 500)  # velocity-x
        vy = np.random.uniform(-500, 500)  # velocity-y
        vz = np.random.uniform(-500, 500)  # velocity-z

        bodies.extend([px, py, pz, vx, vy, vz])

        # Mass range 1e20 to 1e25
        masses.append(np.random.uniform(1e20, 1e25))

    return (np.array(bodies), np.array(masses))
