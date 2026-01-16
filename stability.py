import numpy as np
import integrators

def stability(y,delta,method,y_old,t0,deriv,masses,radii,tf,h,contact_state,toler):
    y_pertubated = y_old + delta
    k = 0
    
    if (method =="rk45"):
        k,t, y_new,contact_state = integrators.rk45(deriv, t0, y_pertubated, tf, h, toler,masses,radii,contact_state,k)
    
    elif (method == "rk4"):
        t, y_new,k,contact_state = integrators.rk4(deriv, t0, y_pertubated, tf, h,masses,radii,contact_state,k)
    
    elif (method == "verlet"):
        t, y_new,k,contact_state = integrators.verlet_step(t0, y_pertubated, masses, tf, h, radii, contact_state, k)
    
    delta_t = y_new - y   

    norm0 = np.linalg.norm(delta)
    normt = np.linalg.norm(delta_t)
    print(delta)

    lambda_t = (1.0/t[1:]) * np.log(normt/norm0)

    print("Collisions in perturbated run:", k)
    
    lambda_total = np.sum(lambda_t)*(1.0/tf)
    
    

    return lambda_total
    
    
    
