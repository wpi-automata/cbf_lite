import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

from cbfs import vanilla_cbf_circle as cbf
from cbfs import vanilla_clf as clf
from dynamics import SimpleDynamics
from sensor import ubiased_noisy_sensor as sensor

# -----------------------------
# QP solve via jaxopt.BoxOSQP
# -----------------------------
def solve_clf_cbf_qp_boxosqp(
    x_estimated: np.ndarray,
    goal: np.ndarray,
    obstacle: np.ndarray,
    safe_radius: float,
    dynamics,
    gamma: float = 1.0,
    alpha: float = 1.0,
    lambda_delta: float = 1000.0,
    u_max: float = 1.0,
    warm_start=None,
):
    """
    Decision: z = [u_x, u_y, delta]
    min ||u||^2 + lambda_delta * delta^2
    s.t.
      LfV + LgV u + gamma V <= delta
      Lfh + Lgh u + alpha h >= 0
      ||u||_inf <= u_max
    Implemented as l <= A z <= u for BoxOSQP.
    """

    x = np.asarray(x_estimated, dtype=float)
    g = np.asarray(goal, dtype=float)
    o = np.asarray(obstacle, dtype=float)

    # --- CLF terms (belief-based) ---
    V = float(clf(x, g))
    L_f_V = float(2.0 * (x - g) @ dynamics.f(x))
    L_g_V = np.asarray(2.0 * (x - g) @ dynamics.g(x), dtype=float).reshape(-1)  # (2,)

    # --- CBF terms (belief-based) ---
    h = float(cbf(x, o, safe_radius))
    L_f_h = float(2.0 * (x - o) @ dynamics.f(x))
    L_g_h = np.asarray(2.0 * (x - o) @ dynamics.g(x), dtype=float).reshape(-1)  # (2,)

    # JAX arrays
    LgV = jnp.asarray(L_g_V, dtype=jnp.float32)
    Lgh = jnp.asarray(L_g_h, dtype=jnp.float32)
    LfV = jnp.asarray(L_f_V, dtype=jnp.float32)
    Lfh = jnp.asarray(L_f_h, dtype=jnp.float32)
    Vj  = jnp.asarray(V, dtype=jnp.float32)
    hj  = jnp.asarray(h, dtype=jnp.float32)

    # Objective: 0.5 z^T Q z + c^T z
    # Want: ||u||^2 + lambda*delta^2  => Q diag = [2,2,2*lambda]
    Q = jnp.diag(jnp.asarray([2.0, 2.0, 2.0 * lambda_delta], dtype=jnp.float32))
    c = jnp.zeros((3,), dtype=jnp.float32)

    # Constraints as l <= A z <= u, z=[u_x,u_y,delta]
    # Box: -u_max <= u_x <= u_max, -u_max <= u_y <= u_max, delta free
    # CLF: LfV + LgV u + gamma V <= delta
    #   => LgV u - delta <= -(LfV + gamma V)
    a_clf = jnp.asarray([LgV[0], LgV[1], -1.0], dtype=jnp.float32)
    b_clf = -(LfV + gamma * Vj)

    # CBF: Lfh + Lgh u + alpha h >= 0
    #   => -(Lgh u) <= (Lfh + alpha h)
    a_cbf = jnp.asarray([-Lgh[0], -Lgh[1], 0.0], dtype=jnp.float32)
    b_cbf = (Lfh + alpha * hj)

    A = jnp.stack([
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),  # u_x
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),  # u_y
        jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32),  # delta
        a_clf,
        a_cbf,
    ], axis=0)

    l = jnp.asarray([-u_max, -u_max, -jnp.inf, -jnp.inf, -jnp.inf], dtype=jnp.float32)
    u = jnp.asarray([ u_max,  u_max,  jnp.inf,  b_clf,   b_cbf  ], dtype=jnp.float32)

    solver = BoxOSQP(tol=1e-6, maxiter=4000, verbose=False)

    # Warm start: pass prior sol.params back in as init_params (if available)
    init_params = warm_start

    sol = solver.run(
        params_obj=(Q, c),
        params_eq=A,          # (we use A as "eq matrix" in the API; constraints are in params_ineq)
        params_ineq=(l, u),   # l <= A z <= u
        init_params=init_params,
    )

    # BoxOSQP returns primal as a tuple: (x, z)
    # x is the decision variable, z is the auxiliary (projected) variable
    x_qp, z_aux = sol.params.primal

    x_qp = np.asarray(x_qp, dtype=float).reshape(-1)  # -> [u_x, u_y, delta]
    u_opt = x_qp[:2]
    delta_opt = float(x_qp[2])

    return u_opt, delta_opt, sol.params


# -----------------------------
# Simulation (same structure)
# -----------------------------
dt = 0.1
T = 100
x_traj = []
u_traj = []

x_true = np.array([-1.5, -1.5], dtype=float)
goal = np.array([2.0, 2.0], dtype=float)

obstacle = np.array([1.0, 1.0], dtype=float)
safe_radius = 0.0

dynamics = SimpleDynamics()

warm = None
for t in range(T):
    x_measured = sensor(x_true, t, std=0.01)
    x_estimated = x_measured

    u_opt, delta_opt, warm = solve_clf_cbf_qp_boxosqp(
        x_estimated=x_estimated,
        goal=goal,
        obstacle=obstacle,
        safe_radius=safe_radius,
        dynamics=dynamics,
        gamma=1.0,
        alpha=1.0,
        lambda_delta=10.0,
        u_max=1.0,
        warm_start=warm,
    )

    # Apply control to true state (add dt; remove dt if you want to match your original exactly)
    x_true = x_true + dt * (dynamics.f(x_true) + dynamics.g(x_true) @ u_opt)

    x_traj.append(x_true.copy())
    u_traj.append(u_opt.copy())

x_traj = np.array(x_traj)

plt.figure(figsize=(6, 6))
plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
plt.scatter(goal[0], goal[1], c="g", marker="*", s=200, label="Goal")
plt.scatter(obstacle[0], obstacle[1], c="r", marker="o", s=200, label="Obstacle")
circle = plt.Circle(obstacle, safe_radius, color="r", fill=False, linestyle="--")
plt.gca().add_patch(circle)

plt.xlim(-2, 3)
plt.ylim(-2, 3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("CLF-CBF QP-Controlled Trajectory (Belief-based) — jaxopt.BoxOSQP")
plt.legend()
plt.grid()
plt.show()
