import numpy as np
import matplotlib.pyplot as plt
import integrators
import body
import Energy
from mpl_toolkits.mplot3d import Axes3D
import csv

G = 6.67430e-11  # Gravitational Constant
N = 100  # No. of objects

# generate bodies with random positions and velocities
bodies_state, masses = body.bodies(N)


def deriv(t, bodies_state):
    # to calculate derivatives for RK4 and RK45
    dydt = np.zeros_like(bodies_state)

    for i in range(N):
        idx = 6 * i
        x, y, z = bodies_state[idx], bodies_state[idx + 1], bodies_state[idx + 2]

        # velocity --> dx/dt
        dydt[idx] = bodies_state[idx + 3]
        dydt[idx + 1] = bodies_state[idx + 4]
        dydt[idx + 2] = bodies_state[idx + 5]

        ax, ay, az = 0, 0, 0
        for j in range(N):
            if i == j:
                continue
            jdx = 6 * j
            dx = bodies_state[jdx] - x
            dy = bodies_state[jdx + 1] - y
            dz = bodies_state[jdx + 2] - z

            r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e4

            # Acceleration = G * m_j / r^3 * vec_r
            ax += G * masses[j] * dx / r**3
            ay += G * masses[j] * dy / r**3
            az += G * masses[j] * dz / r**3

        dydt[idx + 3] = ax
        dydt[idx + 4] = ay
        dydt[idx + 5] = az

    return dydt


def main():
    t0 = 0.0
    tf = 8640000.0  # time duration for simulation in seconds
    h = 1000.0  # step size: 1000 seconds (simulation takes large jumps)
    toler = 1e-3

    y0 = np.array(bodies_state, dtype=float)

    # selection of integrator
    method = "verlet"  # options: "verlet", "rk4", "rk45"

    print(f"Running simulation using {method.upper()}...")

    if method == "rk45":
        t, y = integrators.rk45(deriv, t0, y0, tf, h, toler)
    elif method == "rk4":
        t, y = integrators.rk4(deriv, t0, y0, tf, h)
    elif method == "verlet":
        # verlet integrator from integrators.py
        t, y = integrators.verlet_step(t0, y0, masses, tf, h, G=G)

    print("Simulation complete. Calculating energy...")

    # to calculate energy using Energy.py
    try:
        Total = Energy.total_energy_drift(y, masses, N)
        plt.figure()
        plt.grid()
        plt.plot(t, Total)
        plt.title(f"Total Energy over Time ({method})")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.savefig(f"energy_{method}.png")  # saving plot for total energy
        print(f"Energy plot saved as energy_{method}.png")  # confirmation message
    except Exception as e:
        print(f"Could not plot energy: {e}")
    scale = 1e9
    E0 = Total[0]
    Ef = Total[-1]
    t0 = t[0]
    tf = t[-1]

    

    max_energy_deviation = np.max(np.abs(Total - E0))

    max_energy_drift_rate = max_energy_deviation / (tf - t0)
    print("Max energy drift rate =", max_energy_drift_rate, "J/s")

    # plotting animation
    scale = 1e9

    print("Generating animation... please wait.")
    import matplotlib.animation as animation

    fig = plt.figure(figsize=(10, 10))

    # 1. Setup the Figure
    ax = fig.add_subplot(111, projection="3d")

    # ... inside animation section ...

    # 1. Get all positions (ignore velocities)
    # Shape: (TimeSteps, N, 3)
    all_positions = y.reshape((len(t), N, 6))[:, :, :3]

    # to calculate the max range in which 90% of bodies lie
    max_range = np.percentile(np.abs(all_positions), 90) / scale

    # setting the camera to focus on main cluster of bodies
    limit = max_range * 0.1

    print(f"Camera Limit set to: {limit} billion meters")

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.set_xlabel("X (1e9 m)")
    ax.set_ylabel("Y (1e9 m)")
    ax.set_zlabel("Z (1e9 m)")
    ax.set_title(f"N-Body Simulation ({method})")

    lines = [ax.plot([], [], [], "-", alpha=0.5)[0] for _ in range(N)]
    dots = [ax.plot([], [], [], "o")[0] for _ in range(N)]

    skip = 50
    num_frames = len(t) // skip

    def update(frame):
        current_index = frame * skip

        state = y[current_index]
        reshaped = state.reshape((N, 6))

        for i in range(N):

            history = y[:current_index:10]
            hist_reshaped = history.reshape((-1, N, 6))

            lines[i].set_data(
                hist_reshaped[:, i, 0] / scale, hist_reshaped[:, i, 1] / scale
            )
            lines[i].set_3d_properties(hist_reshaped[:, i, 2] / scale)

            px = reshaped[i, 0] / scale
            py = reshaped[i, 1] / scale
            pz = reshaped[i, 2] / scale

            dots[i].set_data([px], [py])
            dots[i].set_3d_properties([pz])

        return lines + dots

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=30, blit=False
    )

    ani.save("nbody_simulation.gif", writer="pillow", fps=20)
    print("Animation saved as nbody_simulation.gif")


if __name__ == "__main__":
    main()
