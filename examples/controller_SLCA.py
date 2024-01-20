import argparse

import control
import fym
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=10)
        x0 = np.vstack((1, 0))
        self.plant = fym.BaseSystem(x0)
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.vstack((0, 1))
        self.C = np.vstack((1, 0)).T
        self.G = np.vstack((1, 1, 1)).T
        self.Q = np.diag([1, 1])
        self.R = 1
        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        posd = np.vstack((0, 0, 0))
        posd_dot = np.vstack((0, 0, 0))
        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        x = self.plant.state
        nu = -self.K @ x
        # u = np.linalg.pinv(self.G) @ nu  #CCA
        u = self.get_SLCA(nu)
        self.plant.dot = self.A @ x + self.B @ self.G @ u
        env_info = {
            "t": t,
            "x": x,
            "nu": nu,
            "u": u,
        }

        return env_info                  
            
            
    def get_SLCA(self, nu):
        x = self.plant.state.ravel()
        z = self.C @ x
        nu = nu.ravel()
        ws = 1
        wdnu = 1
        pM = vM = 1 
        pm = vm = -1
        uM = 5
        um = -5
        h1 = pM - x[0] + (((max(0,x[1]))**2)/(2*um))  
        h2 = -pm + x[0] - (((max(0,x[1]))**2)/(2*uM))  
        h3 = vM - x[1]
        h4 = -vm + x[1]
        round_h1 = np.array([-1 , max(0,x[1])])
        round_h2 = np.array([1 , -max(0,x[1])])
        round_h3 = np.array([0 , -1])
        round_h4 = np.array([0 , 1])
        f_x = np.vstack([x[1] , 0])
        G_x = np.array([[0 , 0, 0 ], [1, 1, 1]])
        gamma = 5   
        """
        s = y[0]
        dnu = y[1]
        u = y[2:]
        """
        cost = lambda y: ws * y[0] ** 2 + wdnu * y[1] ** 2 + y[2:].T @ y[2:]
        # bnds = ()
        eq1 = lambda y: self.G @ y[2:] - (nu + y[1])
        ineq1 = lambda y: y[0] - (z * y[1])
        ineq2 = lambda y: y[0]
        ineq3 = lambda y: y[2] + 5
        ineq4 = lambda y: y[3] + 5
        ineq5 = lambda y: y[4] + 5
        ineq6 = lambda y: round_h1@f_x+round_h1@G_x@y[2:]+gamma*h1
        ineq7 = lambda y: round_h2@f_x+round_h2@G_x@y[2:]+gamma*h2
        ineq8 = lambda y: round_h3@f_x+round_h3@G_x@y[2:]+gamma*h3
        ineq9 = lambda y: round_h4@f_x+round_h4@G_x@y[2:]+gamma*h4
        cons = (
            {"type": "eq", "fun": eq1},
            {"type": "ineq", "fun": ineq1},
            {"type": "ineq", "fun": ineq2},
            {"type": "ineq", "fun": ineq3},
            {"type": "ineq", "fun": ineq4},
            {"type": "ineq", "fun": ineq5},
            {"type": "ineq", "fun": ineq6},
            {"type": "ineq", "fun": ineq7},
            {"type": "ineq", "fun": ineq8},
            {"type": "ineq", "fun": ineq9},

        )
        result = minimize(cost, np.zeros(5), method="SLSQP", constraints=cons)
        u_star = result.x[2:]
        return np.vstack(u_star)


def run():
    env = MyEnv()
    flogger = fym.Logger("data.h5")

    env.reset()
    try:
        while True:
            env.render()

            done, env_info = env.step()
            flogger.record(env=env_info)

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data.h5")["env"]

    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    ax = axes[0, 0]
    ax.plot(data["t"], data["x"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$x_1$, m")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["x"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$x_2$, m/s")

    ax = axes[2, 0]
    ax.plot(data["t"], data["nu"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$\nu$")

    ax.set_xlabel("Time, s")

    ax = axes[0, 1]
    ax.plot(data["t"], data["u"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$u_1$")

    ax = axes[1, 1]
    ax.plot(data["t"], data["u"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$u_2$")

    ax = axes[2, 1]
    ax.plot(data["t"], data["u"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$u_3$")

    ax.set_xlabel("Time, s")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    plt.show()


def main(args):
    if args.only_plot:
        plot()
        return
    else:
        run()

        if args.plot:
            plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    args = parser.parse_args()
    main(args)