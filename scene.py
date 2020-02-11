from atmosphere import Atmosphere
import data_read


class Scene2D:
    """Class of the simple 2-dimensional math model of the action world."""
    def __init__(self, xml_rocket: str, xml_traj: str):
        """
        :param xml_rocket: rocket's XML data file's path
        :param xml_traj: trajectory's XML data file's path
        """
        self.atmo = Atmosphere()
        self.rocket = data_read.create_rocket(self.atmo, xml_rocket, xml_traj)
        self.is_rocket_calc = False

    # Public
    def sim(self, xml_optim=None):
        """
        Starts the simulation.
        :param xml_optim: optimizer's XML data file's path
        :type xml_optim: str
        """
        if xml_optim is not None:
            self.rocket.optimize_traj(data_read.init_optim_range(xml_optim))
        self.rocket.start()
        self.is_rocket_calc = True

    def show_atmosphere(self):
        """Shows atmosphere's parameters graphs."""
        self.atmo.show()

    def show_rocket_results(self):
        if self.is_rocket_calc is True:
            self.rocket.show_results()
        else:
            print("Warning: the rocket wasn't been calculated!")
