import numpy as np

def heat_fd(ti: float, tf: float,
            xl: float, xr: float,
            h: float, k: float) -> np.ndarray:
    """
        Solve the heat equation using the
        forwards difference method.

        Parameters:

            ti: initial time value
            tf: final time value
            xl: x coordinate of left point
            xr: x coordinate of right point
            h: x step
            k: time step
        
        Returns:

            heat values at every (t,x) point
    """

    initial_condition = lambda x: np.sin(2 * np.pi * x) ** 2
    boundary_value_left = lambda t: 0 * t
    boundary_value_right = lambda t: 0 * t
    diffusion_coefficient = 1

    dt = k
    t = np.arange(ti, tf, dt)
    t_count = len(t)
    dx = h
    x = np.arange(xl, xr, dx)
    x_count = len(x)

    sigma = diffusion_coefficient * dt / (dx ** 2)

    A = np.zeros((x_count, x_count), dtype=np.float32)
    for i in range(1, x_count - 1):
        A[i, i - 1] = sigma
        A[i,i] = 1 - 2 * sigma
        A[i, i + 1] = sigma
    
    w = np.zeros((x_count, t_count), dtype=np.float32)
    w[:,0] = initial_condition(x)
    
    t = 0
    for j in range(t_count - 1):
        w[:, j + 1] = A.dot(w[:, j])
        w[0, j + 1] = boundary_value_left(t + dt)
        w[x_count - 1, j + 1] = boundary_value_right(t + dt)
        t += dt

    return w

print(heat_fd(0, 1, 0, 1, 0.1, 0.002))