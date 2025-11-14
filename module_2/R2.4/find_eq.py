import numpy as np

def find_eq(T, a, z=None):
    # find_eq - find equilibrium position along z axis
    #
    # Usage:
    # z = find_eq(T, a)
    # z = find_eq(T, a, initial_guess)
    # where T = T-matrix, a = field vector.

    newton = True

    z_precision = 1e-4
    short_distance = 1e-5
    zpoints = 45

    if z is None:
        z = 0

    z_old = z + 10 * z_precision

    if newton:
        # Newton's method
        while abs(z - z_old) > z_precision:
            a2 = translate_z(a, z)
            a3 = translate_z(a2, short_distance)

            p = np.dot(T, a2)
            f1 = force_z(a2, p)

            p = np.dot(T, a3)
            f2 = force_z(a3, p)

            dz = short_distance * f1 / (f1 - f2)

            z_old = z
            z = z + dz

    else:
        # Bisection method

        # Need initial guess
        size_T = T.shape
        radius = size_T[0] / (2 * np.pi)

        z = np.linspace(-radius, 3 * radius, zpoints)

        fz = np.zeros(z.shape)

        for nz in range(zpoints):
            a2 = translate_z(a, z[nz])

            p = np.dot(T, a2)
            fz[nz] = force_z(a2, p)

            if fz[nz] < 0:
                z1 = z[nz - 1]
                z2 = z[nz]
                f1 = fz[nz - 1]
                f2 = fz[nz]
                break

        if f1 == 0:
            z = z1
            return z

        if nz == zpoints - 1:
            raise ValueError('Starting points for bisection not found')

        # Now the actual bisection search
        while z2 - z1 > z_precision:
            z3 = (z1 + z2) / 2

            a2 = translate_z(a, z3)

            p = np.dot(T, a2)
            f3 = force_z(a2, p)

            if f3 == 0:
                z = z3
                break
            if f1 * f3 < 0:
                z1 = z3
                f1 = f3
            else:
                z2 = z3
                f2 = f3

        z = z3

    return z

