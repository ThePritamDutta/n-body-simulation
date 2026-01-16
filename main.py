from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import integrators
import body
import Energy
from mpl_toolkits.mplot3d import Axes3D
import h5py
import time
import matplotlib.animation as animation

G = 6.67430e-11  # Gravitational Constant
N = 2

# Place them on the X-axis, moving toward each other
bodies_state = np.array([
    -5.0e8, 0.0, 0.0,  500.0,  80.0, 0.0,
     5.0e8, 0.0, 0.0, -500.0, -80.0, 0.0
])

masses = np.array([1e25, 1e25])
radii = np.array([6e6, 6e6])

def deriv(t, bodies_state):
    state = bodies_state.reshape((N, 6))
    dydt = np.zeros((N, 6))
    dydt[:, :3] = state[:, 3:]
    acc = integrators.get_acc(bodies_state, masses)
    dydt[:, 3:] = acc
    return dydt.ravel()

def main(radii, masses):
    start_time1 = time.perf_counter()
    t0 = 0.0
    tf = 3600 * 24 * 12
    h = 1# Increased step slightly for faster solving; set to 1 for high precision
    toler = 1e-5
    y0 = bodies_state

    method = "verlet"
    start_time = time.perf_counter()
    print(f"Running simulation using {method.upper()}...")

    if method == "rk45":
        t, y = integrators.rk45(deriv, t0, y0, tf, h, toler, masses, radii)
    elif method == "rk4":
        t, y = integrators.rk4(deriv, t0, y0, tf, h, masses, radii)
    elif method == "verlet":
        t, y = integrators.verlet_step(t0, y0, masses, tf, h, radii)
    print("1st Simultion done")
    
    delta = 1e-8
    y_old = y0
    
    lambda_stability = stability.stability(y,delta,method,y_old,t0,deriv,masses,radii,tf,h)
    print("2nd simulation done")
    print("The stability of the system:",lambda_stability)

    print("Simulation complete. Calculating energy...")
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"Runtime = {runtime:.3f} seconds")

    # ================= ENERGY & MOMENTUM =================
    Total = Energy.total_energy(y, masses, N)
    E0 = Total[0]
    Energy_drift = (Total - E0) / abs(E0)

    plt.figure(); plt.grid(); plt.plot(t, Energy_drift)
    plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
    plt.savefig(f"energy_drift_rel_{method}.png")

    Angular_Momentum_Vector, Angular_drift = Energy.Angular_momentum(y, masses, N)
    plt.figure(); plt.grid(); plt.plot(t, Angular_drift)
    plt.xlabel("Time"); plt.ylabel("Angular Momentum Drift")
    plt.savefig(f"Angular_Momentum_Drift_{method}.png")

    # ================= ANIMATION =================
    print("Generating animation... please wait.")
    scale = 1e9 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    limit = 1.2 # Zoomed in slightly to see the 0.5e9 distance better
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.set_xlabel("X (1e9 m)"); ax.set_ylabel("Y (1e9 m)"); ax.set_zlabel("Z (1e9 m)")
    ax.set_title(f"{N} Body Simulation using ({method})")

    radii_plot = np.array(radii) / 5e7
    lines = [ax.plot([], [], [], "-", alpha=0.3, linewidth=max(1, radii_plot[i] * 5))[0] for i in range(N)]
    dots = [ax.scatter([], [], [], s=(radii_plot[i] * 3000), alpha=0.9) for i in range(N)]

    skip = 200
    num_frames = max(1, len(t) // skip)
    history = y[::skip].reshape(-1, N, 6)


    def update(frame):
        current_index = min(frame * skip, len(y) - 1)

        reshaped = y[current_index].reshape((N, 6))
        
        # Update positions and trails
        for i in range(N):
            px, py, pz = reshaped[i, :3] / scale
            dots[i]._offsets3d = (np.array([px]), np.array([py]), np.array([pz]))
            
            # Update trails (minimal change to include history)
            lines[i].set_data(history[:frame+1, i, 0]/scale,
                  history[:frame+1, i, 1]/scale)
            lines[i].set_3d_properties(history[:frame+1, i, 2]/scale)

            

        return lines + dots

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)

    writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    ani.save(f"{N}_body_simulation_using_{method}.mp4", writer=writer)
    print("Animation saved successfully.")
    
    plt.close(fig)
    print("Total simulation time:", time.perf_counter() - start_time1, "seconds")
    with h5py.File(f"{N}simulation_with_{method}.h5", "w") as f:
        
        f.create_dataset("time", data=t)
        f.create_dataset("state", data=y)
        f.create_dataset("masses", data=masses)
        f.create_dataset("radii", data=radii)
        
        f.attrs["Runtime"] = runtime
        f.attrs["Total simulation time(experimental)"] = Total_simulation_time
        f.attrs["Max_Energy Drift"] = max_energy_drift_rate
        f.attrs["Max_Angular Momentum drift"] = max_angular_drift_rate
        f.attrs["Total No of Collision"] = k
        

        f.attrs["method"] = method
        f.attrs["stepsize_h"] = h
        f.attrs["total simulation time(theoritical)"] = tf
        f.attrs["N"] = len(masses)
    print("H5 file saved")

if __name__ == "__main__":
    main(radii, masses)
