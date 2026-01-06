import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"
import integrators
import body
import Energy
from mpl_toolkits.mplot3d import Axes3D
import csv
import time
import matplotlib.animation as animation
from numba import njit

G = 6.67430e-11  # Gravitational Constant
N = 100 # No. of objects


# generate bodies with random positions and velocities
bodies_state, masses = body.bodies(N)

@njit
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

        acc = integrators.get_acc(bodies_state, masses, N, G=6.67430e-11, eps=1e4)
        

        dydt[idx + 3] = acc[3*i]
        dydt[idx + 4] = acc[3*i+1]
        dydt[idx + 5] = acc[3*i+2]

    return dydt


def main():
    start_time1 = time.perf_counter()
    t0 = 0.0
    tf = 3600*24*50  # time duration for simulation in seconds
    h = 1000  # step size: 6000 seconds (simulation takes large jumps)
    toler = 1e-5
    
    
    y0 = bodies_state
    print(bodies_state)
    print(masses)
   

    # selection of integrator
    method = "verlet"  # options: "verlet", "rk4", "rk45"
    start_time = time.perf_counter()
    print(f"Running simulation using {method.upper()}...")

    if method == "rk45":
        t, y = integrators.rk45(deriv, t0, y0, tf, h, toler)
    elif method == "rk4":
        t, y = integrators.rk4(deriv, t0, y0, tf, h)
    elif method == "verlet":
        # verlet integrator from integrators.py
        t, y = integrators.verlet_step(t0, y0, masses, tf, h, G=G)

    print("Simulation complete. Calculating energy...")
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    print(f"Runtime = {runtime:.3f} seconds")
    # to calculate energy using Energy.py
    try:
        Total = Energy.total_energy(y, masses, N)
        plt.figure()
        plt.grid()
        plt.plot(t, Total)
        plt.title(f"Total Energy Drift over Time ({method})")
        plt.xlabel("Time")
        plt.ylabel("Energy Drift")
        plt.savefig(f"energydrift_{method}.png")  # saving plot for total energy
        print(f"Energy plot saved as energy drift_{method}.png")  # confirmation message
    except Exception as e:
        print(f"Could not plot energy: {e}")

    # plotting animation
    scale = 1e9
    E0 = Total[0]
    Ef = Total[-1]
    t0 = t[0]
    tf = t[-1]

    

    max_energy_deviation = np.max(np.abs(Total - E0))

    max_energy_drift_rate = max_energy_deviation / (tf - t0)
    
    print("Max energy drift rate =", max_energy_drift_rate, "J/s")

    Angular_Momentum_Vector,Angular_drift = Energy.Angular_momentum(y,masses,N)
    Angular_drift.shape == (len(t),)
    max_angular_drift = np.max(np.abs(Angular_drift))

    max_angular_drift_rate = max_angular_drift / (tf - t0)
    
    print(f"The maximum angular drift rate of the system using the method {method} is",max_angular_drift_rate)
    
    
    plt.figure()
    plt.grid()
    plt.plot(t,Angular_Momentum_Vector)
    plt.title(f"Total Angular Momentum over time t using ({method})")
    plt.xlabel("Time(t)")
    plt.ylabel("Angular_Momentum")
    plt.savefig(f"Angular_Momentum({method}).png")
    print(f"Energy plot saved as angular momentum {method}.png")
    
    
    
    plt.figure()
    plt.grid()
    plt.plot(t,Angular_drift)
    plt.title(f"Total Angular Momentum Drift over time t using ({method}) for N = {N}")
    plt.xlabel("Time(t)")
    plt.ylabel("Angular Momentum drift")
    plt.savefig(f"Angular_Momentum_Drift USING {method} f.png")
    print(f"Energy plot saved as angular momentum drift_{method}.png")
    

    print("Generating animation... please wait.")
    

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
    limit = 4

    print(f"Camera Limit set to: {limit} billion meters")

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.set_xlabel("X (1e9 m)")
    ax.set_ylabel("Y (1e9 m)")
    ax.set_zlabel("Z (1e9 m)")
    ax.set_title(f"{N} Body Simulation using ({method})")

    lines = [ax.plot([], [], [], "-", alpha=0.5)[0] for _ in range(N)]
    dots = [ax.plot([], [], [], "o")[0] for _ in range(N)]

    skip = 25
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

    ani.save(f"{N}_body_simulation using {method}.mp4", writer="ffmpeg", fps=10)
    print(f"Animation saved as {N} body_simulation {method}.mp4")
    start_time2 = time.perf_counter()
    Total_simulation_time = start_time2 - start_time1
    print("The total simulation time:",Total_simulation_time,"seconds")


if __name__ == "__main__":
    main()

