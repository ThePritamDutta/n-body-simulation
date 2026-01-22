import numpy as np

def bodies(N, G=6.67430e-11, mass_mode="random"):
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    masses = np.zeros(N)
    radii = np.zeros(N)

    
    R0 = 2.0e8    
    V0 = 500.0    

   
    dirs = np.random.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]

    r = np.random.random(N)**(1/3)  
    positions = dirs * r[:, None] * R0

    velocities = np.random.normal(0.0, V0, size=(N, 3))

    for i in range(N):
        radii[i] = np.random.uniform(1e6, 5e8)

 
    if mass_mode == "segregation_test":
        M0 = 1e25
       
        masses[:15] = 1.0 * M0
      
        masses[15:25] = 2.0 * M0
      
        masses[25:30] = 4.0 * M0

     
    else:
        for i in range(N):
            masses[i] = np.random.uniform(1e20, 1e25)

   
    U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]

            r_ij = np.sqrt(dx*dx + dy*dy + dz*dz)
            U -= G * masses[i] * masses[j] / r_ij

   
  
    K = 0.0
    for i in range(N):
        v2 = np.sum(velocities[i]**2)
        K += 0.5 * masses[i] * v2

    
    scale = np.sqrt(np.abs(U) / (2.0 * K))
    velocities *= scale

 
    total_mass = np.sum(masses)
    vcm = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    velocities -= vcm

 
    state = np.zeros((N, 6))
    state[:, :3] = positions
    state[:, 3:] = velocities

    return state.ravel(), masses, radii
