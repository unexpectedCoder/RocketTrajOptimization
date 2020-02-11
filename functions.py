# Gas dynamics (gd) and other helper functions
import numpy as np


# Constants
g = 9.80665             # Gravity acceleration, m/s^2
r_earth = 6.371e6       # Earth radius, m


def flow_dens(k: float, lam: float):
    """
    Calculates the impulse flux density.
    :param k: adiabatic index of flow
    :param lam: dimensionless velocity
    :return: impulse flux density
    """
    return (1 + lam**2) * np.power(1 - (k-1)/(k+1) * lam**2, 1/(k-1))


def gd_pi(k: float, lam: float):
    """
    Calculates gas dynamics pi-function (pressure ratio).
    :param k: adiabatic index
    :param lam: dimensionless flow velocity
    :return: gas dynamics pi-function
    """
    return np.power(1 - (k-1)/(k+1) * lam**2, k / (k-1))


def gd_lambda(k: float, mach: float):
    """
    Calculates dimensionless velocity depending on the Mach number.
    :param k: adiabatic index
    :param mach: Mach number
    :return: dimensionless velocity
    """
    return mach * np.sqrt(0.5 * (k + 1) / (1 + 0.5 * (k - 1) * mach**2))


def gd_lambda_pi(k: float, gd_pi_val: float):
    """
    Calculates dimensionless velocity depending on the GD pressure function.
    :param k: adiabatic index
    :param gd_pi_val: value of the GD pressure ratio
    :return: dimensionless velocity
    """
    return np.sqrt((k+1)/(k-1) * (1 - np.power(gd_pi_val, (k-1)/k)))


def k_reactive(lam: float):
    """
    Calculates coefficient of the nozzle's reactivity.
    :param lam: dimensionless velocity "lam"
    """
    if lam > 0:
        return 0.5 * (lam + 1 / lam)
    return None


def cx(h: float, mach: float):
    """
    Calculates approx coefficient of the aerodynamic resistance Cx (drag) using Mach number.
    :param h: flight height
    :param mach: Mach number
    :return: drag coefficient
    """
    if h < 80000:
        if 0 <= mach <= 0.8:
            return 0.29
        if mach <= 1.068:
            return mach - 0.51
        if mach < 6:
            return 0.091 + 0.5 / mach
        return 0.17
    return 0
