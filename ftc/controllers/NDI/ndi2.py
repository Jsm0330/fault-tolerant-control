import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm
from numpy import cos, sin, tan


class NDIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        """ outer-loop control
        Objective: horizontal position (x, y) tracking control
        States:
            pos[0:2]: horizontal position
            posd[0:2]: desired horizontal position
        """
        xo, xod = pos[0:2], posd[0:2]
        xo_dot, xod_dot = vel[0:2], posd_dot[0:2]
        eo, eo_dot = xo - xod, xo_dot - xod_dot
        Ko1 = 0.5 * np.diag((3, 1))
        Ko2 = 0.5 * np.diag((3, 2))

        # outer-loop virtual control input
        nuo = (-Ko1 @ eo - Ko2 @ eo_dot) / env.plant.g
        angd = np.vstack((nuo[1], -nuo[0], 0))
        # angd = np.deg2rad(np.vstack((0, 0, 0)))  # to regulate Euler angle

        """ inner-loop control
        Objective: vertical position (z) and angle (phi, theta, psi) tracking control
        States:
            pos[2]: vertical position
            posd[2]: desired vertical position
            ang: Euler angle
            angd: desired Euler angle
        """
        J = env.plant.J
        xi_dot = np.vstack((vel[2], omega))
        M = np.vstack(
            (
                env.plant.g,
                omega[1] * omega[2] * ((J[1][1] - J[0][0]) / J[0][0]),
                omega[0] * omega[2] * ((J[2][2] - J[0][0]) / J[1][1]),
                omega[0] * omega[1] * ((J[0][0] - J[1][1]) / J[2][2]),
            )
        )

        E = np.diag(
            (
                float(-np.cos(ang[0]) * np.cos(ang[1]) / env.plant.m),
                1 / J[0][0],
                1 / J[1][1],
                1 / J[1][1],
            )
        )
        A = -np.linalg.inv(E) @ M
        B = np.linalg.inv(E)

        xi = np.vstack((pos[2], ang))
        xid = np.vstack((posd[2], angd))
        xi_dot = np.vstack((vel[2], omega))
        xid_dot = np.vstack((posd_dot[2], 0, 0, 0))
        ei = xi - xid
        ei_dot = xi_dot - xid_dot
        Ki1 = 5 * np.diag((5, 10, 50, 10))
        Ki2 = 1 * np.diag((5, 10, 50, 10))
        nui = np.vstack((-Ki1 @ ei - Ki2 @ ei_dot))
        u = A + B @ nui

        ctrls = np.linalg.pinv(env.plant.mixer.B) @ u
        controller_info = {
            "posd": posd,
            "angd": angd,
            "ang": ang,
        }

        return ctrls, controller_info
