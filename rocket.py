from atmosphere import Atmosphere
from engine import Engine
from htypes import *
from traj import *
import functions as func

from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt


class Rocket2D:
    """A simple model of the air ballistics rocket."""
    def __init__(self, atmo, s_conds, params, constraints, engine, parts, traj):
        """
        :param atmo: model of the atmosphere
        :type atmo: Atmosphere
        :param s_conds: starting conditions for the rocket
        :type s_conds: StartConds
        :param params: rocket's parameters
        :type params: Parameters
        :param constraints: constraints for the rocket's math model
        :type constraints: Constraints
        :param engine: rocket's engine
        :type engine: Engine
        :param parts: rocket's compartments array
        :type parts: list[Compartment]
        :param traj: rocket's trajectory
        :type traj: Traj
        """
        if s_conds.coords.dim != 2:
            print("Error in <Rocket>: coordinates dimension is not equal to 2!")
            exit(-1)
        self.atmo = copy(atmo)
        self.s_conds = copy(s_conds)
        self.params = copy(params)
        self.constraints = copy(constraints)
        self.engine = copy(engine)
        self.parts = copy(parts)
        self.traj = copy(traj)
        # Others
        self.mass = 0.0
        for part in self.parts:
            self.mass += part.mass
        self.maneuver = Maneuver()
        # Timing parameters of this model
        self.state = State(self.s_conds.t, self.s_conds.coords, self.s_conds.V, self.s_conds.mu, self.s_conds.Theta)
        self.result = Result2D(np.array(self.state.t),
                               np.array(self.state.coords.x[0]), np.array(self.state.coords.x[1]),
                               np.array(self.state.V),
                               np.array(self.state.mu),
                               np.array(self.state.Theta))
        # For optimizations
        self.optim_params = OptimParams()       # Optimal parameters init by default
        # Events dictionary
        self.ev_dict = {'time': self.__event_time, 'theta': self.__event_theta, 'hit_ground': self.__event_hit_ground,
                        'max_dist': self.__event_max_dist, 'max_height': self.__event_max_height, 'march_height': self.__event_march_height,
                        'mu': self.__event_mu}

    def __copy__(self):
        rocket = Rocket2D(self.atmo, self.s_conds, self.params, self.constraints, self.engine, self.parts, self.traj)
        rocket.mass = self.mass
        rocket.maneuver = copy(self.maneuver)
        rocket.state = copy(self.state)
        rocket.result = copy(self.result)
        if self.optim_params is not None:
            rocket.optim_params = copy(self.optim_params)
        return rocket

    def __str__(self):
        output = f"Rocket:" \
                 f"\n {self.atmo}" \
                 f"\n\t Total mass, kg: {self.mass}" \
                 f"\n {self.s_conds}" \
                 f"\n {self.state}" \
                 f"\n {self.constraints}" \
                 f"\n {self.params}" \
                 f"\n {self.engine}" \
                 f"\n\t Compartments:"
        for part in self.parts:
            output += f"\n {part}"
        output += f"\n {self.traj}"
        return output

    # Public
    def optimize_traj(self, optim_range: list):
        """
        Optimizes the rocket's trajectory using dual annealing algorithm.
        :param optim_range: list of min and max values of the optimization parameters (use list of zip pairs)
        """
        # Optimization
        optim = dual_annealing(self.__start_optimization, optim_range, maxiter=100, args=[Traj(self.__exclude_man_dive())])
        self.optim_params.set(optim.x[0], optim.x[1], optim.x[2])
        print("\nMessage from <Rocket2D>: trajectory is optimized successfully.\n")
        print(self.optim_params)

        self.reset()

    def start(self, trajectory=None, rough=False, max_step=0.2):
        """
        Launches the rocket.
        :param trajectory: rocket's trajectory
        :type trajectory: Traj
        :param rough: use rough hit ground event
        :type rough: bool
        :param max_step: max step for the ODE's solver
        """
        traj = copy(self.traj) if trajectory is None else copy(trajectory)
        # Checking and preparing
        traj_names = traj.names()
        for name in traj_names:
            if name not in traj_types:
                print("Error in <Rocket2D: invalid trajectory part's name!")
                exit(-1)
        # Fly
        self.engine.on(False)
        is_first = True
        is_first_man = True

        for part in traj.parts:
            if self.state.coords.x[1] > self.constraints.max_height:
                break

            if isinstance(part, float) is False:
                events = self.__form_events(part.events)
            else:
                events = None
                self.state.Theta = part
                continue

            if part.name == traj_types[0]:                                              # const theta
                if self.engine.is_run() is False:
                    if self.engine.on(True) is False:
                        print("Warning from <Rocket2D>: engine has no more operating modes!")

                if is_first:
                    is_first = False
                    t_span = np.array([0.0, self.optim_params.k_t_start * part.t])
                else:
                    t_span = np.array([self.state.t, self.state.t + 300.0])

                self.__move_const_theta(t_span, max_step / 3, events, rough=rough)

                if self.engine.is_run() and self.state.mu > self.engine.break_mu():
                    self.engine.off()
                traj.next()
                continue
            if part.name == traj_types[1]:                                              # maneuver
                t_span = np.array([self.state.t, self.state.t + 50.0])
                if is_first_man:
                    is_first_man = False
                    self.maneuver.set(self.state.Theta, self.optim_params.Theta_start, part.overload_share)
                else:
                    self.maneuver.set(self.state.Theta, part.Theta_end, part.overload_share)

                self.__move_maneuver(t_span, max_step / 5, events, rough=rough)

                traj.next()
                continue
            if part.name == traj_types[2]:                                              # passive
                if self.engine.is_run():
                    self.engine.off()
                t_span = np.array([self.state.t, self.state.t + 500.0])

                self.__move_passive(t_span, max_step, events, rough=rough)

                traj.next()
                continue
            if part.name == traj_types[3]:                                              # dive
                if self.state.coords.x[0] > self.constraints.max_dist and self.state.coords.x[1] > 0.0:
                    t_span = np.array([self.state.t, self.state.t + 50.0])
                    self.maneuver.set(self.state.Theta, part.Theta_end, part.overload_share)

                    if rough is False:
                        print(events)
                    self.__move_maneuver(t_span, max_step / 5, events)

                    self.state.Theta = -np.pi / 2
                    # To finish
                    t_span = np.array([self.state.t, self.state.t + 100.0])

                    self.__move_passive(t_span, max_step / 2, events)
                break       # Dive is always the latest trajectory's part

    def update(self, t: np.ndarray, y: np.ndarray):
        """
        Updates current rocket's state.
        :param t: time's array, s
        :param y: ODE solution's results
        """
        self.state.set(t[-1], Coordinates(np.array([y[1, -1], y[2, -1]])), y[0, -1], y[3, -1], y[4, -1])
        self.result.append(t, y[1], y[2], y[0], y[3], y[4])

    def reset(self):
        """Returns the rocket's state to the initial state."""
        self.engine.reset()
        self.state = State(self.s_conds.t, self.s_conds.coords, self.s_conds.V, self.s_conds.mu, self.s_conds.Theta)
        self.result = Result2D(np.array(self.state.t),
                               np.array(self.state.coords.x[0]), np.array(self.state.coords.x[1]),
                               np.array(self.state.V),
                               np.array(self.state.mu),
                               np.array(self.state.Theta))

    def show_results(self):
        """Shows calculated rocket's trajectory and some others parameters."""
        # Kinematic and other parameters
        plt.figure("Rocket's Parameters")
        # V(t)
        plt.subplot(2, 2, 1)
        plt.plot(self.result.t, self.result.V, color='black')
        plt.ylim(0)
        plt.xlabel('$t$, с')
        plt.ylabel('$V$, м/с')
        plt.grid(True)
        # x(t)
        plt.subplot(2, 2, 2)
        plt.plot(self.result.t, self.result.x * 1e-3, color='black')
        plt.ylim(0)
        plt.xlabel('$t$, с')
        plt.ylabel('$x$, км')
        plt.grid(True)
        # h(t)
        plt.subplot(2, 2, 3)
        plt.plot(self.result.t, self.result.y * 1e-3, color='black')
        plt.ylim(0)
        plt.xlabel('$t$, с')
        plt.ylabel('$y$, км')
        plt.grid(True)
        # mu(t)
        plt.subplot(2, 2, 4)
        plt.plot(self.result.t, self.result.mu, color='black')
        plt.ylim(0)
        plt.xlabel('$t$, с')
        plt.ylabel('$\\mu$')
        plt.grid(True)

        # Trajectory
        plt.figure("Rocket's Trajectory")
        plt.plot(self.result.x * 1e-3, self.result.y * 1e-3, color='black')
        plt.ylim(0)
        plt.xlabel('$x$, км')
        plt.ylabel('$h$, км')
        plt.grid(True)

        plt.show()

    # Private
    def __start_optimization(self, x: list, traj: Traj):
        # Makes single calculation for the optimization.
        self.reset()
        self.optim_params.set(x[0], x[1], x[2])
        self.start(trajectory=traj, rough=True, max_step=1.0)
        # For maximization
        return -self.state.coords.x[0]

    def __exclude_man_dive(self):
        traj = []
        for part in self.traj.parts:
            if part.name != 'man' and part.name != 'dive':
                traj.append(part)
            elif part.name == 'man':
                traj.append(part.Theta_end)
        return traj

    # High-level calculation functions
    def __move_const_theta(self, t_span: np.ndarray, max_step: float, events: list = None, rough=False):
        y0 = self.state.to_ndarray_xy()
        if events is None:
            sol = solve_ivp(self.__ode_const_theta, t_span, y0, args=[rough], max_step=max_step)
        else:
            sol = solve_ivp(self.__ode_const_theta, t_span, y0, args=[rough], max_step=max_step, events=events)
        self.update(sol.t, sol.y)

    def __move_passive(self, t_span: np.ndarray, max_step: float, events: list = None, rough=False):
        y0 = self.state.to_ndarray_xy()
        if events is None:
            sol = solve_ivp(self.__ode_passive, t_span, y0, args=[rough], max_step=max_step)
        else:
            sol = solve_ivp(self.__ode_passive, t_span, y0, args=[rough], max_step=max_step, events=events)
        self.update(sol.t, sol.y)

    def __move_maneuver(self, t_span: np.ndarray, max_step: float, events: list = None, rough=False):
        y0 = self.state.to_ndarray_xy()
        if events is None:
            if self.engine.is_run():
                sol = solve_ivp(self.__ode_active_man, t_span, y0, args=[rough], max_step=max_step)
            else:
                sol = solve_ivp(self.__ode_passive_man, t_span, y0, args=[rough], max_step=max_step)
        else:
            if self.engine.is_run():
                sol = solve_ivp(self.__ode_active_man, t_span, y0, args=[rough], max_step=max_step, events=events)
            else:
                sol = solve_ivp(self.__ode_passive_man, t_span, y0, args=[rough], max_step=max_step, events=events)
        self.update(sol.t, sol.y)

    # Low-level calculation functions
    def __ode_const_theta(self, t, y, rough):
        # ODE system of the rocket's motion. It's prepared for using in scipy.integrate.
        v, x, h, mu, theta = y
        # Helper values
        K = 0.5 * self.atmo.k * self.atmo.p_h0
        M = v / self.atmo.sonic(h)
        Pi = self.atmo.p(h) / self.atmo.p_h0
        Pi_out = self.engine.p_outlet() / self.engine.p0()
        lam_out = func.gd_lambda_pi(self.engine.k(), Pi_out)
        A = self.engine.eta() / (1 - mu)
        B = self.engine.eta() * (1 - Pi) /\
            ((1 - mu) * (func.flow_dens(self.engine.k(), lam_out) *
                         self.engine.p0() / self.atmo.p_h0 - 1))
        C = -0.5 * self.params.icx * func.cx(h, M) * K * Pi * M ** 2 / (self.params.q_m * (1 - mu))
        D = -np.sin(theta)
        # ODE system
        return [func.g * (A + B + C + D),
                v * np.cos(theta),
                v * np.sin(theta),
                func.g * self.engine.eta() / self.engine.imp1(),
                0.0]

    def __ode_passive(self, t, y, rough):
        # ODE system of the rocket's passive motion. It's prepared for using in scipy.integrate.
        v, x, h, mu, theta = y
        # Helper variables
        K = 0.5 * self.atmo.k * self.atmo.p_h0
        M = v / self.atmo.sonic(h)
        Pi = self.atmo.p(h) / self.atmo.p_h0
        C = -0.5 * self.params.icx * func.cx(h, M) * K * Pi * M ** 2 / (self.params.q_m * (1 - mu))
        D = -np.sin(theta)
        # ODE system
        return [func.g * (C + D),
                v * np.cos(theta),
                v * np.sin(theta),
                0.0,
                -func.g * np.cos(theta) / v]

    def __ode_active_man(self, t, y, rough):
        # ... ODE system for active maneuver (engine is running)
        v, x, h, mu, theta = y
        # Helper values
        K = 0.5 * self.atmo.k * self.atmo.p_h0
        M = v / self.atmo.sonic(h)
        Pi = self.atmo.p(h) / self.atmo.p_h0
        Pi_out = self.engine.p_outlet() / self.engine.p0()
        lam_out = func.gd_lambda_pi(self.engine.k(), Pi_out)
        A = self.engine.eta() / (1 - mu)
        B = self.engine.eta() * (1 - Pi) / \
            ((1 - mu) * (func.flow_dens(self.engine.k(), lam_out) *
                         self.engine.p0() / self.atmo.p_h0 - 1))
        C = -0.5 * self.params.icx * func.cx(h, M) * K * Pi * M ** 2 / (self.params.q_m * (1 - mu))
        D = -np.sin(theta)
        # ODE system
        dtheta = self.__theta_program(v)
        return [func.g * (A + B + C + D),
                v * np.cos(theta),
                v * np.sin(theta),
                func.g * self.engine.eta() / self.engine.imp1(),
                dtheta]

    def __ode_passive_man(self, t, y, rough):
        # ... ODE system of the rocket's maneuver when the engine isn't running
        v, x, h, mu, theta = y
        # Helper variables
        K = 0.5 * self.atmo.k * self.atmo.p_h0
        M = v / self.atmo.sonic(h)
        Pi = self.atmo.p(h) / self.atmo.p_h0
        C = -0.5 * self.params.icx * func.cx(h, M) * K * Pi * M ** 2 / (self.params.q_m * (1 - mu))
        D = -np.sin(theta)
        # ODE system
        dtheta = self.__theta_program(v)
        return [func.g * (C + D),
                v * np.cos(theta),
                v * np.sin(theta),
                0.0,
                dtheta]

    def __theta_program(self, v: float):
        theta0, theta_end, overload_share = self.maneuver.get()
        if theta0 > theta_end:
            return -overload_share * self.__calc_allowable_dtheta(v)
        return overload_share * self.__calc_allowable_dtheta(v)

    def __calc_allowable_dtheta(self, velo: float):
        # Calculates allowable dTheta/dt using max ny
        if velo > 0.0:
            return self.constraints.max_ny * func.g / velo
        return 0.0

    # Events
    def __form_events(self, ev_names: list):
        if ev_names == [None]:
            return None
        return [self.ev_dict[ev] for ev in ev_names]

    def __event_time(self, t, y, rough):
        return t - (self.traj.get().t + self.state.t)
    __event_time.direction = 1
    __event_time.terminal = True

    def __event_theta(self, t, y, rough):
        return y[4] - self.maneuver.Theta_end
    __event_theta.direction = 0
    __event_theta.terminal = True

    def __event_max_height(self, t, y, rough):
        return y[2] - self.constraints.max_height
    __event_max_height.direction = 1
    __event_max_height.terminal = True

    def __event_march_height(self, t, y, rough):
        return self.optim_params.h_march - y[2]
    __event_march_height.direction = 1
    __event_march_height.terminal = True

    def __event_hit_ground(self, t, y, rough):
        if rough:
            return y[2] - 2000.0
        return y[2] - 200.0
    __event_hit_ground.direction = -1
    __event_hit_ground.terminal = True

    def __event_max_dist(self, t, y, rough):
        return y[1] - self.constraints.max_dist
    __event_max_dist.direction = 1
    __event_max_dist.terminal = True

    def __event_mu(self, t, y, rough):
        return y[3] - self.engine.break_mu()
    __event_mu.direction = 1
    __event_mu.terminal = True
