from numpy import rad2deg


traj_types = ['const_theta', 'man', 'passive', 'dive']


class TrajPart:
    """Trajectory part's parameters."""
    def __init__(self, name: str, t=None, theta=None, overload_share=None, events=None):
        """
        :param name: trajectory's part's name (type)
        :type name: str
        :param t: motion's time interval, s
        :type t: float
        :param theta: angle of inclination of the velocity vector to the horizon at the end of the part, rad
        :type theta: float
        :param overload_share: transverse overload share
        :type overload_share: float
        :param events: list of events for interruption the ODE system's solution
        :type events: list
        """
        self.name = name
        self.t = t if t is not None else None
        self.Theta_end = theta if theta is not None else None
        self.overload_share = overload_share if overload_share is not None else None
        self.events = events.copy() if events is not None else None

    def __copy__(self):
        return TrajPart(self.name, self.t, self.Theta_end, self.overload_share, self.events)

    def __str__(self):
        output = "\t part's parameters" \
                 f"\n\t - name (type): {self.name}"
        if self.t is not None:
            output += f"\n\t - motion's time interval, s: {self.t}"
        if self.Theta_end is not None:
            output += f"\n\t - angle of inclination to the horizon at the end, deg: {rad2deg(self.Theta_end)}"
        if self.overload_share is not None:
            output += f"\n\t - transverse overload share: {self.overload_share}"
        if self.events is not None:
            output += f"\n\t - events: {self.events}"
        return output


class Traj:
    """Trajectory's type."""
    def __init__(self, parts: list):
        self.parts = parts.copy()
        self.indx = 0

    def __copy__(self):
        traj = Traj(self.parts)
        traj.indx = self.indx
        return traj

    def __str__(self):
        output = "\t Trajectory:"
        for part in self.parts:
            output += f"\n{str(part)}"
        return output

    def next(self):
        """Switches trajectory's part."""
        if self.indx < len(self.parts):     # '==' condition appears after the latest trajectory's part
            self.indx += 1
        else:
            print("Warning from <Traj>: trajectory has no more parts!")

    def get(self):
        """Returns current part of trajectory."""
        return self.parts[self.indx]

    def names(self):
        """Returns the list of trajectories names (types)."""
        return [part.name for part in self.parts if isinstance(part, float) is False]
