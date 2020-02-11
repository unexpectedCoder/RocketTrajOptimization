# Module of helper types
from copy import copy
import numpy as np


class Compartment:
    def __init__(self, name: str, mass: float):
        """
        :param name: name of rocket's compartment
        :param mass: mass of rocket's compartment, kg
        """
        if mass < 0.0:
            print("Error in <Compartment>: compartment mass cannot be negative!")
            exit(-1)
        self.name = name
        self.mass = mass

    def __copy__(self):
        return Compartment(self.name, self.mass)

    def __str__(self):
        return f"\t Compartment '{self.name}':" \
               f"\n\t - mass, kg: {self.mass}"


class Constraints:
    """Constraints for the rocket's math model."""
    def __init__(self, max_dist: float, max_height: float, max_ny: float):
        """
        :param max_dist: maximum horizontal distance of flight, m
        :param max_height: maximum height of flight, m
        :param max_ny: maximum transverse overload value
        """
        self.max_dist = max_dist
        self.max_height = max_height
        self.max_ny = max_ny

    def __copy__(self):
        return Constraints(self.max_dist, self.max_height, self.max_ny)

    def __str__(self):
        return f"\t Constraints:" \
               f"\n\t - maximum horizontal distance of flight, km: {self.max_dist * 1e-3}" \
               f"\n\t - maximum height of flight, km: {self.max_height * 1e-3}" \
               f"\n\t - maximum transverse overload value: {self.max_ny}"


class Coordinates:
    """The N dimension coordinates."""
    out_acc = 3		# Accuracy of the output

    def __init__(self, x: np.ndarray):
        """
        :param x: list of the rocket's coordinates, m
        """
        self.x = x.copy()
        self.dim = len(x)

    def __copy__(self):
        return Coordinates(self.x)

    def __str__(self):
        out = f'coordinates ({round(self.x[0] * 1e-3, self.out_acc)}'
        if len(self.x) > 1:
            for x in self.x[1:]:
                out += f'; {round(x * 1e-3, self.out_acc)}'
        return out + f') km in {self.dim}D space'

    def change_output_accuracy(self, acc: int):
        """
        Sets the output accuracy (number of decimal places).
        :param acc: output accuracy
        """
        if acc < 0:
            self.out_acc = 3
        else:
            self.out_acc = acc

    def update(self, new_x):
        """
        Updates coordinates using the list of the new coordinates.
        :param new_x: new coordinates
        :type new_x: np.ndarray[float]
        """
        if len(new_x) != self.dim:
            print('Error in <Coordinates>: coordinates list length is not equal to space dimension!')
            exit(-1)
        self.x = new_x.copy()


class Maneuver:
    """Helper class for rocket's maneuver description."""
    def __init__(self):
        self.Theta_start = 0.0
        self.Theta_end = 0.0
        self.overload_share = 0.0

    def __copy__(self):
        mp = Maneuver()
        mp.set(self.Theta_start, self.Theta_end, self.overload_share)
        return mp

    def set(self, theta_start: float, theta_end: float, overload_share: float):
        """
        Sets the maneuver parameters.
        :param theta_start: starting angle of inclination to the horizon, rad
        :param theta_end: finish angle of inclination to the horizon, rad
        :param overload_share: transverse overload's share
        """
        self.Theta_start = theta_start
        self.Theta_end = theta_end
        self.overload_share = overload_share

    def get(self):
        """Returns tuple of the maneuver's parameters."""
        return self.Theta_start, self.Theta_end, self.overload_share


class OptimParams:
    """Class for saving optimization results."""
    out_acc = 3     # Output accuracy

    def __init__(self, start_theta=np.pi/4, k_t_start=0.5, h_march=4e4):
        """
        :param start_theta: angle of inclination of the velocity vector to the horizon, rad
        :type start_theta: float
        :param k_t_start: time of the starting part ratio
        :type k_t_start: float
        :param h_march: height of the beginning march operating mode, m
        :type h_march: float
        """
        self.Theta_start = start_theta
        self.k_t_start = k_t_start
        self.h_march = h_march
        self.n_params = 3

    def __copy__(self):
        return OptimParams(self.Theta_start, self.k_t_start, self.h_march)

    def __str__(self):
        return f" Optimal parameters:" \
               f"\n - angle of inclination of the velocity vector to the horizon, deg: {round(np.rad2deg(self.Theta_start), self.out_acc)}" \
               f"\n - part of time of the starting part: {round(self.k_t_start, self.out_acc)}" \
               f"\n - height of the beginning march operating mode, km: {round(self.h_march * 1e-3, self.out_acc)}"

    def set(self, start_theta: float, k_t: float, h_march: float):
        """
        Sets the optimal parameters of the rocket's trajectory.
        :param start_theta: optimal angle of inclination of the velocity vector to the horizon, rad
        :param k_t: optimal time of the starting part ratio
        :param h_march: optimal height of the beginning march operating mode, m
        """
        self.Theta_start = start_theta
        self.k_t_start = k_t
        self.h_march = h_march


class Parameters:
    """Parameters for the rockets."""
    def __init__(self, icx: float, beta: float, q_m: float, lam_l: float):
        """
        :param icx: shape factor
        :param beta: mass excellence
        :param q_m: middle load, Pa
        :param lam_l: lengthening of corpus
        """
        self.icx = icx
        self.beta = beta
        self.q_m = q_m
        self.lamL = lam_l

    def __copy__(self):
        return Parameters(self.icx, self.beta, self.q_m, self.lamL)

    def __str__(self):
        return f"\t Parameters:" \
               f"\n\t - aerodynamics shape factor: {self.icx}" \
               f"\n\t - mass excellence (beta coefficient): {self.beta}" \
               f"\n\t - middle load, Pa: {self.q_m}" \
               f"\n\t - lengthening of corpus: {self.lamL}"


class Result2D:
    """Describes the solution results for the 2D-space model (X, Y: distance, height) of the scene."""
    out_acc = 3       # Output accuracy

    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, v: np.ndarray, mu: np.ndarray, theta: np.ndarray):
        """
        :param t: array float of the time, s
        :param x: array of the x-coordinate (distance), m
        :param y: array of the y-coordinate (height), m
        :param v: array of the velocity, m/s
        :param mu: array of the relative propellant mass
        :param theta: array of the angle of inclination of the velocity vector to the horizon, rad
        """
        self.t = t.copy()
        self.x = x.copy()
        self.y = y.copy()
        self.V = v.copy()
        self.mu = mu.copy()
        self.Theta = theta.copy()

    def __copy__(self):
        return Result2D(self.t, self.x, self.y, self.V, self.mu, self.Theta)

    def __str__(self):
        return " Results of the solution:" \
               f"\n - time of the rocket's flight, s: {round(self.t[-1], self.out_acc)}" \
               f"\n - max velocity, m/s: {round(np.max(self.V), self.out_acc)}" \
               f"\n - finish distance, km: {round(self.x[-1] * 1e-3, self.out_acc)}" \
               f"\n - max height, km: {round(np.max(self.y) * 1e-3, self.out_acc)}" \
               f"\n - total relative propellant mass: {round(self.mu[-1], self.out_acc)}"

    def append(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, v: np.ndarray, mu: np.ndarray, theta: np.ndarray):
        """
        Appends ODE system of the rocket's motion  solution results to self data.
        :param t: array of the time, s
        :param x: array of the x-coordinate (distance), m
        :param y: array of the y-coordinate (height), m
        :param v: array of the velocity, m/s
        :param mu: array of the relative propellant mass
        :param theta: array of the angle of inclination of the velocity vector to the horizon, rad
        """
        self.t = np.hstack([self.t, t])
        self.x = np.hstack([self.x, x])
        self.y = np.hstack([self.y, y])
        self.V = np.hstack([self.V, v])
        self.mu = np.hstack([self.mu, mu])
        self.Theta = np.hstack([self.Theta, theta])


class StartConds:
    """Starting conditions of the rocket's ballistics task."""
    def __init__(self, time: float, coords: Coordinates, velo: float, mu: float, theta: float):
        """
        :param coords: starting coordinates, m
        :param velo: starting velocity, m/s
        :param mu: starting relative propellant mass
        :param theta: starting angle of inclination of the velocity vector to the horizon, rad
        :param time: starting time
        """
        self.t = time
        self.coords = copy(coords)
        self.V = velo
        self.mu = mu
        self.Theta = theta

    def __copy__(self):
        return StartConds(self.t, self.coords, self.V, self.mu, self.Theta)

    def __str__(self):
        return f"\t Starting conditions:" \
               f"\n\t - starting time, s: {self.t}" \
               f"\n\t - {self.coords}" \
               f"\n\t - velocity, m/s: {self.V}" \
               f"\n\t - relative propellant mass: {self.mu}" \
               f"\n\t - angle of inclination to the horizon, deg: {np.rad2deg(self.Theta)}"


class State:
    """Current state of the rocket."""
    out_acc = 3

    def __init__(self, time: float, coords: Coordinates, velo: float, mu: float, theta: float):
        """
        :param time: current time, s
        :param coords: current coordinates of the rocket, m
        :param velo: current rocket's velocity, m/s
        :param mu: current relative propellant mass
        :param theta: current angle of inclination of the velocity vector to the horizon, rad
        """
        self.t = time
        self.coords = copy(coords)
        self.V = velo
        self.mu = mu
        self.Theta = theta

    def __copy__(self):
        return State(self.t, self.coords, self.V, self.mu, self.Theta)

    def __str__(self):
        return "\tCurrent state:" \
               f"\n\t - time, s: {self.t}" \
               f"\n\t - {self.coords}" \
               f"\n\t - velocity, m/s: {round(self.V, self.out_acc)}" \
               f"\n\t - relative propellant mass: {round(self.mu, self.out_acc)}" \
               f"\n\t - angle of inclination to the horizon, deg: {round(np.rad2deg(self.Theta), self.out_acc)}"

    def set(self, time: float, coords: Coordinates, velo: float, mu: float, theta: float):
        """
        Sets parameters of the new state.
        :param time: current time, s
        :param coords: current coordinates of the rocket, m
        :param velo: current rocket's velocity, m/s
        :param mu: current relative propellant mass
        :param theta: current angle of inclination of the velocity vector to the horizon, rad
        """
        self.t = time
        self.coords = copy(coords)
        self.V = velo
        self.mu = mu
        self.Theta = theta

    def to_ndarray_xy(self) -> np.ndarray:
        """Returns State as NumPy's array."""
        return np.array([self.V, self.coords.x[0], self.coords.x[1], self.mu, self.Theta])
