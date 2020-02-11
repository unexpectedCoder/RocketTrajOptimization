from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


class Atmosphere:
    """It is a standard atmosphere model (GOST 4401-81)."""
    def __init__(self, source_file_path='data/atmosphere.txt', spl='\t'):
        self.is_read = False                                                # Flag: source file was read successfully
        self.file_path = source_file_path                                   # Data file path
        self.spliter = spl
        self.k = 1.4                                                        # Adiabatic index
        self.h_table, temp, p, rho = self.__read_datafile(source_file_path, spl)
        self.p_h0 = p[0]                                                    # Pressure on surface
        self.h_range = [self.h_table[0], self.h_table[-1]]                  # Atmosphere height's range
        self.T, self.p, self.rho = self.__interpolate_tables(temp, p, rho)  # Interpolation funcs

    def __copy__(self):
        return Atmosphere(self.file_path, self.spliter)

    def __str__(self):
        return f'\t Atmosphere: standard atmosphere according to GOST 4401-81 ' \
               f'from {self.h_range[0] * 1e-3} to {self.h_range[-1] * 1e-3} km'

    def __read_datafile(self, path, spl):
        try:
            h, temp, p, rho = [], [], [], []
            with open(path, 'r') as f:
                for line in f:
                    buf = [float(v) for v in line.split(spl)]
                    h.append(buf[0])
                    temp.append(buf[1])
                    p.append(buf[2] * 1e5)
                    rho.append(buf[3])
            h_table = np.array(h, float)
            self.is_read = True
            return h_table, np.array(temp), np.array(p), np.array(rho)
        except IOError:
            self.is_read = False
            print(f'{IOError}\nError: file is not found!')
            exit(-1)

    def __interpolate_tables(self, temp: np.ndarray, p: np.ndarray, rho: np.ndarray):
        # Temperature, pressure, density
        return interp1d(self.h_table, temp, kind='cubic'),\
               interp1d(self.h_table, p, kind='linear'),\
               interp1d(self.h_table, rho, kind='linear')

    def sonic(self, h: float):
        """
        Calculates atmosphere's sonic speed using pressure (p) and density (rho).
        @param h: height of flight
        @type h: float
        @return: sonic speed
        """
        if h < 0.0 or h > 1e5:
            return 1e-6
        return np.sqrt(self.k * self.p(h) / self.rho(h))

    def show(self):
        """
        Shows tabled and interpolated atmosphere data depending on height:
        pressure p(h), temperature T(h) and density rho(h).
        """
        h_step = 100
        h = np.arange(self.h_range[0], self.h_range[1], h_step)     # Numeric heights (not table)
        temp_table = np.array([self.T(height) for height in self.h_table])
        temp = self.T(h)
        p_table = np.array([self.p(height) for height in self.h_table])
        p = self.p(h)
        rho_table = np.array([self.rho(height) for height in self.h_table])
        rho = self.rho(h)

        plt.figure('Atmosphere')

        plt.subplot(3, 1, 1)
        plt.plot(h * 1e-3, temp, color='blue')
        plt.plot(self.h_table * 1e-3, temp_table, 'x', color='r')
        plt.xlabel('$H$, км')
        plt.ylabel('$T$, К')
        plt.xlim(self.h_range[0] * 1e-3, self.h_range[-1] * 1e-3)
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(h * 1e-3, p * 1e-5, color='blue')
        plt.plot(self.h_table * 1e-3, p_table * 1e-5, 'x', color='r')
        plt.xlabel('$H$, км')
        plt.ylabel('$p \\cdot 10^{-5}$, Па')
        plt.xlim(self.h_range[0] * 1e-3, self.h_range[-1] * 1e-3)
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(h * 1e-3, rho, color='blue')
        plt.plot(self.h_table * 1e-3, rho_table, 'x', color='r')
        plt.xlabel('$H$, км')
        plt.ylabel('$\\rho$, кг/м$^3$')
        plt.xlim(self.h_range[0] * 1e-3, self.h_range[1] * 1e-3)
        plt.grid(True)

        plt.show()
