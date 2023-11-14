import numpy as np
from enum import Enum
from copy import deepcopy

import pinocchio as pin


class InterpType(Enum):
    POS = 0
    ROT = 1
    SE3 = 2


class Interpolator:
    def __init__(self, interp_type=InterpType.POS):
        self.i = 0
        self.traj_t = []
        self._done = True
        self._start = None
        self._goal = None

        self.interp_type = interp_type

        assert interp_type in InterpType
        if interp_type == InterpType.POS:
            self.interp_fn = self.lerp
        elif interp_type == InterpType.ROT:
            self.interp_fn = self.slerp
        elif interp_type == InterpType.SE3:
            self.interp_fn = self.se3_lerp

    def __len__(self):
        return len(self.traj)

    def init(
        self,
        start,
        goal,
        steps,
    ):
        if self.interp_type == InterpType.SE3:
            isinstance(start, pin.SE3)
            isinstance(goal, pin.SE3)
        self._start = start
        self._goal = goal
        self.traj_t = np.linspace(0, 1, steps)
        self.i = 0
        self._done = False

    def next(self):
        if self.i < len(self.traj_t) - 1:
            self.i += 1
        else:
            self._done = True
        return self.interp_fn(self._start, self._goal, self.traj_t[self.i])

    @property
    def done(self):
        return self._done

    @staticmethod
    def lerp(start, end, t):
        return (1 - t) * start + t * end

    @staticmethod
    def slerp(start, end, t):
        """Spherical Linear intERPolation."""
        return (end * start.inverse()) ** t * start

    @staticmethod
    def se3_lerp(start, end, t):
        """SE3 linear interpolation."""
        n = start * pin.exp6(t * pin.log6(start.inverse() * end))
        return n


if __name__ == "__main__":
    start = pin.SE3.Identity()
    end = pin.SE3.Identity()
    end.translation = np.array([1, 0, 1])
    end.rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )  # rotate('x', pi/2)
    interp = Interpolator(interp_type=InterpType.SE3)
    interp.init(start, end, 1000)
    assert np.allclose(interp.interp_fn(start, end, interp.traj_t[0]), start)
    assert np.allclose(interp.interp_fn(start, end, interp.traj_t[-1]), end)
