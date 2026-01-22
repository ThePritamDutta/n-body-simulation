import numpy as np
import matplotlib.pyplot as plt
import integrators
import body
import Energy
import h5py
import time
import matplotlib.animation as animation
import stability
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

TEST_MASS_SEGREGATION = True
N_RUNS = 3

G = 6.67430e-11
DEFAULT_N = 50


def main():
    BASE_SEED = 12345
    for run_id in range(N_RUNS if TEST_MASS_SEGREGATION else 1):
        seed = BASE_SEED + run_id
        
        print(f"\n=== RUN {run_id+1} ===")
        np.random.seed(seed)

        if TEST_MASS_SEGREGATION:
            N = 30
            bodies_state, masses, radii = body.bodies(
                N, mass_mode="segregation_test"
            )

            radii = (masses / np.max(masses))**(1/3) * 5e8
            integrators.ENABLE_COLLISIONS = False
            print("Mass segregation test mode: collisions OFF")
        else:
            N = DEFAULT_N
            bodies_state, masses, radii = body.bodies(N)

        def deriv(t, bodies_state):
            state = bodies_state.reshape((N, 6))
            dydt = np.zeros((N, 6))
            dydt[:, :3] = state[:, 3:]
            acc = integrators.get_acc(bodies_state, masses)
            dydt[:, 3:] = acc
            return dydt.ravel()

        t0 = 0.0
        tf = 3600 * 24 * (460 if TEST_MASS_SEGREGATION else 120)
        h = 1000
        toler = 1e-5
        y0 = bodies_state

        method = "verlet" if TEST_MASS_SEGREGATION else "rk45"
        print(f"Running simulation using {method.upper()}")

        contact_state = [[False]*N for _ in range(N)]
        k = 0

        start_time = time.perf_counter()

        if method == "rk45":
            t, y, k, contact_state = integrators.rk45(
                deriv, t0, y0, tf, h, toler,
                masses, radii, contact_state, k
            )
        elif method == "rk4":
            t, y, k, contact_state = integrators.rk4(
                deriv, t0, y0, tf, h,
                masses, radii, contact_state, k
            )
        else:
            t, y, k, contact_state = integrators.verlet_step(
                t0, y0, masses, tf, h,
                radii, contact_state, k
            )

        runtime = time.perf_counter() - start_time
        print("Simulation done")

        tail = y[int(0.8 * len(y)):]
        r_stack = []

        for snap in tail:
            snap = snap.reshape(N, 6)
            pos = snap[:, :3]
            cm = np.average(pos, axis=0, weights=masses)
            r_stack.append(np.linalg.norm(pos - cm, axis=1))

        r_avg = np.mean(r_stack, axis=0)

        m_max = np.max(masses)
        m_min = np.min(masses)

        print("\nTime-averaged mean radius (last 20%, CM-relative):")
        print("Heavy  :", np.mean(r_avg[masses == m_max]))
        print("Medium :", np.mean(
            r_avg[(masses < m_max) & (masses > m_min)]
        ))
        print("Light  :", np.mean(r_avg[masses == m_min]))

        r_char = np.median(r_avg)
        evap_frac = np.sum(
            r_avg[masses == m_min] > 3.0 * r_char
        ) / np.sum(masses == m_min)

        print("Light-body evaporation fraction:", evap_frac)

        Total = Energy.total_energy(y, masses, N)
        E0 = Total[0]
        Energy_drift = (Total - E0) / abs(E0)

        plt.figure()
        plt.grid()
        plt.plot(t, Energy_drift)
        plt.xlabel("Time")
        plt.ylabel("Relative Energy Drift")
        plt.savefig(f"energy_drift_rel_{method}_run{run_id+1}.png")
        plt.close()

        _, Angular_drift = Energy.Angular_momentum(y, masses, N)

        plt.figure()
        plt.grid()
        plt.plot(t, Angular_drift)
        plt.xlabel("Time")
        plt.ylabel("Angular Momentum Drift")
        plt.savefig(f"Angular_Momentum_Drift_{method}_run{run_id+1}.png")
        plt.close()

        lambda_stability = stability.stability(
            y, 1e-8, method, y0, t0,
            deriv, masses, radii,
            tf, h, contact_state, toler
        )
        print("Stability λ =", lambda_stability)

        scale = 1e9
        R_view = 6.0 * np.median(r_avg) / scale

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-R_view, R_view)
        ax.set_ylim(-R_view, R_view)
        ax.set_zlim(-R_view, R_view)
        ax.view_init(elev=20, azim=45)
        ax.set_title("Mass Segregation and Evaporation")

        radii_plot = radii / np.max(radii)

        colors = []
        for mi in masses:
            if mi == m_max:
                colors.append("red")
            elif mi > m_min:
                colors.append("orange")
            else:
                colors.append("blue")

        dots = [
            ax.scatter([], [], [], s=3000*radii_plot[i],
                       c=colors[i], alpha=0.9)
            for i in range(N)
        ]

        trails = [
            ax.plot([], [], [], '-', lw=1, alpha=0.3,
                    color=colors[i])[0]
            for i in range(N)
        ]

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='orange', markersize=10),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='blue', markersize=10),
        ]
        ax.legend(handles=legend_elements)

        skip = 25
        history = y[::skip].reshape(-1, N, 6)
        R_evap = 3.0 * np.median(r_avg)

        def update(frame):
            frame = min(frame, len(history) - 1)
            pos = history[frame, :, :3]
            cm = np.average(pos, axis=0, weights=masses)

            for i in range(N):
                rel_pos = (pos[i] - cm) / scale
                dots[i]._offsets3d = ([rel_pos[0]],
                                      [rel_pos[1]],
                                      [rel_pos[2]])

                r_now = np.linalg.norm(pos[i] - cm)
                if r_now > R_evap:
                    dots[i].set_alpha(0.15)
                    dots[i].set_sizes([700])
                else:
                    dots[i].set_alpha(0.9)
                    dots[i].set_sizes([3000 * radii_plot[i]])

                start = max(0, frame - 10)
                trail_segment = history[start:frame+1, i, :3].copy()
                for j, hist_idx in enumerate(range(start, frame+1)):
                    h_cm = np.average(
                        history[hist_idx, :, :3],
                        axis=0, weights=masses
                    )
                    trail_segment[j] -= h_cm

                trails[i].set_data(trail_segment[:, 0]/scale,
                                   trail_segment[:, 1]/scale)
                trails[i].set_3d_properties(
                    trail_segment[:, 2]/scale
                )

            return dots + trails

        ani = animation.FuncAnimation(
            fig, update, frames=len(history), interval=40
        )
        ani.save(
            f"{N}_body_run_{run_id+1}_{method}.mp4",
            writer=animation.FFMpegWriter(fps=20)
        )
        plt.close(fig)

        plt.figure()
        plt.hist(r_avg[masses == m_max], bins=10, alpha=0.7)
        plt.hist(r_avg[(masses < m_max) & (masses > m_min)],
                 bins=10, alpha=0.7)
        plt.hist(r_avg[masses == m_min], bins=10, alpha=0.7)
        plt.xlabel("CM-Relative Radius")
        plt.ylabel("Count")
        plt.savefig(f"radial_distribution_run{run_id+1}.png")
        plt.close()

        plt.figure()
        plt.boxplot([
            r_avg[masses == m_max],
            r_avg[(masses < m_max) & (masses > m_min)],
            r_avg[masses == m_min]
        ])
        plt.ylabel("CM-Relative Radius")
        plt.savefig(f"radius_boxplot_run{run_id+1}.png")
        plt.close()

        with h5py.File(
            f"{N}_simulation_run_{run_id+1}_{method}.h5", "w"
        ) as f:
            f.create_dataset("time", data=t)
            f.create_dataset("state", data=y)
            f.create_dataset("masses", data=masses)
            f.create_dataset("radii", data=radii)
            f.attrs["Runtime"] = runtime
            f.attrs["Total No of Collision"] = k
            f.attrs["Stability_lambda"] = lambda_stability
            f.attrs["method"] = method


if __name__ == "__main__":
    main()
