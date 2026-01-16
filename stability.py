import numpy as np
import integrators

def stability(y,delta,method,y_old,t0,deriv,masses,radii,tf,h,contact_state):
    y_pertubated = y_old + delta
    k = 0
    
    if (method =="rk45"):
        k,t, y_new = integrators.rk45(deriv, t0, y_pertubated, tf, h, toler,masses,radii,k)
    
    elif (method == "rk4"):
        t, y_new,k = integrators.rk4(deriv, t0, y_pertubated, tf, h,masses,radii,k)
    
    elif (method == "verlet"):
        t, y_new,k = integrators.verlet_step(t0, y_pertubated, masses, tf, h, radii, contact_state, k)
    
    delta_t = y_new - y   

    norm0 = np.linalg.norm(delta)
    normt = np.linalg.norm(delta_t)
    print(delta)

    lambda_t = (1.0/t[1:]) * np.log(normt/norm0)

    print("Collisions in perturbed run:", k)
    
    lambda_total = np.sum(lambda_t)*(1.0/tf)
    
    print("2nd Simulation Done")

    return lambda_total
    
    
    