import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import integrators
import body
import Energy
import time
import csv  # Added for data export
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# Physics Constants
# --------------------------------------------------
G = 6.67430e-11
N = 100                
EPS = 1e4             

# --------------------------------------------------
# Initial Conditions
# --------------------------------------------------
bodies_state, masses = body.bodies(N, seed=42)

def deriv(t, y):
    """
    Standard state-space derivative. 
    y is structured as [x1, y1, z1, vx1, vy1, vz1, ... xN, yN, zN, vN, vN, vN]
    """
    dydt = np.zeros_like(y)

    # dx/dt = v
    dydt[0::6] = y[3::6]
    dydt[1::6] = y[4::6]
    dydt[2::6] = y[5::6]

    # dv/dt = a 
    acc = integrators.get_acc(y, masses, G=G, epsilon=EPS)
    dydt[3::6] = acc[0::3]
    dydt[4::6] = acc[1::3]
    dydt[5::6] = acc[2::3]

    return dydt

def main():
    # Setup timing (100 days)
    t0 = 0.0
    tf = 5e6
    h = 300.0               
    toler = 1e-6               

    y0 = bodies_state.copy()

    # --- SAVE INITIAL CONDITIONS TO CSV ---
    print("Saving initial conditions to initial_conditions.csv...")
    with open('initial_conditions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header row
        writer.writerow(['Body_ID', 'Mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        
        for i in range(N):
            idx = i * 6
            row = [
                i + 1, 
                masses[i],
                y0[idx], y0[idx+1], y0[idx+2],     # Positions
                y0[idx+3], y0[idx+4], y0[idx+5]    # Velocities
            ]
            writer.writerow(row)
    print("CSV saved successfully.")

    # Simulation execution
    method = "verlet" 
    print(f"\nStarting {method.upper()} solver...")

    start_timer = time.perf_counter()

    if method == "rk45":
        t, y = integrators.rk45(deriv, t0, y0, tf, h, toler)
    elif method == "rk4":
        t, y = integrators.rk4(deriv, t0, y0, tf, h)
    elif method == "verlet":
        t, y = integrators.verlet_step(t0, y0, masses, tf, h, G=G, epsilon=EPS)
    else:
        raise ValueError("Check the method string!")

    runtime = time.perf_counter() - start_timer
    print(f"Done! Runtime: {runtime:.3f} seconds")

    # --------------------------------------------------
    # Diagnostics (Energy & Momentum)
    # --------------------------------------------------
    print("\nProcessing Energy and Angular Momentum...")
    
    Total = Energy.total_energy_drift(y, masses, N, eps=EPS)

    plt.figure()
    plt.plot(t, Total)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Energy Drift")
    plt.title(f"Energy Stability ({method})")
    plt.grid(True)
    plt.savefig(f"energydrift_{method}.png")
    plt.close()

    # Angular Momentum checks
    Angular_Momentum_Vector, Angular_drift = Energy.Angular_momentum(y, masses, N)

    plt.figure()
    plt.plot(t, Angular_Momentum_Vector)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Momentum Magnitude")
    plt.title(f"Angular Momentum ({method})")
    plt.grid(True)
    plt.savefig(f"Angular_Momentum_{method}.png")
    plt.close()

    plt.figure()
    plt.plot(t, Angular_drift)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Angular Momentum Drift")
    plt.grid(True)
    plt.savefig(f"Angular_Momentum_Drift_{method}.png")
    plt.close()

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    print("Rendering animation...")
    scale = 1e9  
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    y_plot = y.reshape((len(t), N, 6))
    
    # Use initial positions to set camera scale
    initial_positions = y_plot[0, :, :3]

# Convert to Gm for visualization
    initial_positions_scaled = initial_positions / scale

# Set limits based on initial configuration
    limit = np.max(np.linalg.norm(initial_positions_scaled, axis=1)) * 1.5


    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.set_xlabel("X (Gm)")
    ax.set_ylabel("Y (Gm)")
    ax.set_zlabel("Z (Gm)")

    lines = [ax.plot([], [], [], "-", alpha=0.4)[0] for _ in range(N)]
    dots = [ax.plot([], [], [], "o")[0] for _ in range(N)]

    skip = max(len(t) // 400, 1)

    def update(frame):
        idx = frame * skip
        for i in range(N):
            hist = y_plot[:idx:10, i, :3] / scale
            if hist.size > 0:
                lines[i].set_data(hist[:, 0], hist[:, 1])
                lines[i].set_3d_properties(hist[:, 2])

            pos = y_plot[idx, i, :3] / scale
            dots[i].set_data(pos[0:1], pos[1:2])
            dots[i].set_3d_properties(pos[2:3])

        return lines + dots

    ani = animation.FuncAnimation(
        fig, update, frames=len(t) // skip, interval=30, blit=False
    )

    ani.save(f"n_body_simulation_{method}.mp4", writer="ffmpeg", fps=15)
    print("MP4 saved.")

if __name__ == "__main__":
    main()