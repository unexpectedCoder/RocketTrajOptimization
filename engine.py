import numpy as np


class Engine:
    """A simple model of a solid propellant rocket engine."""
    def __init__(self, eta, imp1, p0, p_outlet, k, break_mu):
        """
        :param eta: weight-to-thrust ratio for appropriate operating modes
        :type eta: np.ndarray
        :param imp1: single impulse for appropriate operating modes, m/s
        :type imp1: np.ndarray
        :param p0: pressure in the combustion chamber for appropriate operating modes, Pa
        :type p0: np.ndarray
        :param p_outlet: nozzle outlet pressure for appropriate operating modes, Pa
        :type p_outlet: np.ndarray
        :param k: adiabatic index of the combustion products for appropriate operating modes
        :type k: np.ndarray
        :param break_mu: relative propellant masses switching operating modes
        :type break_mu: np.ndarray
        """
        if compare_pairs_len_eq([eta, imp1, p0, p_outlet, k, break_mu]) is False:
            print("Error in <Engine>: lengths of arguments are not equal to each other!")
            exit(-1)
        self.__eta = eta.copy()
        self.__I_1 = imp1.copy()
        self.__p0 = p0.copy()
        self.__p_outlet = p_outlet.copy()
        self.__k = k.copy()
        self.__mu_break = break_mu.copy()
        # Timing values
        self.__is_run = False         # Does engine run?
        self.__mode = 0               # Current operating mode
        # Others
        self.__modes = len(eta)       # Amount of engine operating modes

    def __copy__(self):
        engine = Engine(self.__eta, self.__I_1, self.__p0, self.__p_outlet, self.__k, self.__mu_break)
        engine.__is_run = self.__is_run
        engine.__mode = self.__mode
        engine.__modes = self.__modes
        return engine

    def __str__(self):
        return f"\t Engine:" \
               f"\n\t - weight-to-thrust ratio: {self.__eta}" \
               f"\n\t - single impulse, m/s: {self.__I_1}" \
               f"\n\t - pressure in the combustion chamber, MPa: {self.__p0 * 1e-6}" \
               f"\n\t - nozzle outlet pressure, MPa: {self.__p_outlet * 1e-6}" \
               f"\n\t - adiabatic index of the combustion products: {self.__k}" \
               f"\n\t - relative propellant masses switching operating modes: {self.__mu_break}" \
               f"\n\t - current operating mode: {self.__mode + 1}" \
               f"\n\t - amount of operating modes: {self.__modes}" \
               f"\n\t - engine is running: {self.__is_run}"

    def reset(self):
        """Returns engine parameters to the initial state."""
        self.__is_run = False
        self.__mode = 0

    def on(self, next_mode: bool) -> bool:
        """
        Turns on the engine, if possible.
        :param next_mode: if it 'True' then engine turns on next operating mode, if possible
        :return: engine is running (True) or not (False)
        """
        if next_mode is True:
            if self.__mode + 1 < self.__modes:
                self.__mode += 1
                self.__is_run = True
            else:
                self.__is_run = False
                print(f"Warning from <Engine>: engine has not more than {self.__modes} operating modes!")
            return self.__is_run
        self.__is_run = True
        return self.__is_run

    def off(self):
        """Turns off the engine."""
        self.__is_run = False

    def is_run(self) -> bool:
        """Returns True, if engine is running."""
        return self.__is_run

    def eta(self) -> float:
        """Returns weight-to-thrust ratio."""
        return self.__eta[self.__mode]

    def imp1(self) -> float:
        """Returns single impulse."""
        return self.__I_1[self.__mode]

    def p0(self) -> float:
        """Returns pressure in the combustion chamber."""
        return self.__p0[self.__mode]

    def p_outlet(self) -> float:
        """Returns nozzle's outlet pressure."""
        return self.__p_outlet[self.__mode]

    def k(self) -> float:
        """Returns adiabatic index of the combustion products."""
        return self.__k[self.__mode]

    def break_mu(self) -> float:
        """Returns relative propellant masses for switching operating modes."""
        return self.__mu_break[self.__mode]


# Functions
def compare_pairs_len_eq(values) -> bool:
    """
    Compares arrays lengths in the list.
    :param values: list of arrays
    :type values: list
    """
    for i in range(len(values) - 1):
        if len(values[i]) != len(values[i + 1]):
            return False
    return True
