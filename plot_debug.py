import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ============================================================
# 1. LOAD DATA
# ============================================================

data = np.load("sim_sinusoidal_GEKF_v_5.0.npz", allow_pickle=True)

figure_size = (12, 9)


x_traj      = data["x_traj"]
x_meas      = data["x_meas"]
x_est       = data["x_est"]
u_traj      = data["u_traj"]
u_nom       = data["u_nom"]
cbf_values  = data["cbf_values"]
x_nom       = data["x_nom"]
t           = data["time"]
covariances = data["covariances"]
kalman_gains = data["kalman_gains"]

# left_lglfh         = data["left_lglfh"]
# right_lglfh        = data["right_lglfh"]
# left_rhs           = data["left_rhs"]
# right_rhs          = data["right_rhs"]

# right_l_f_h  = data["right_l_f_h_full"]
# right_l_f_2_h = data["right_l_f_2_h_full"]
# left_l_f_h    = data["left_l_f_h_full"]
# left_l_f_2_h  = data["left_l_f_2_h_full"]

# # =====================================
# # Debug Figure: Kalman Gains
# # =====================================

# plt.figure(figsize=figure_size)

# plt.scatter(x_meas[:, 0], x_meas[:, 1], color="green", marker="o", s=1.0, alpha=0.5, label="Measurements")
# plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="True trajectory")
# plt.plot(x_est[:, 0], x_est[:, 1], color="orange", label="Estimated trajectory")
# plt.plot(x_nom[:, 0], x_nom[:, 1], "black", label="Nominal trajectory")
# plt.axhline(y=5.0, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
# plt.axhline(y=-5.0, color="red", linestyle="dashed", linewidth=1)
# plt.xlabel("x [m]"); plt.ylabel("y [m]")
# plt.title(f"2D Trajectory")
# plt.legend(); plt.grid()

# # =====================================
# # Debug Figure: Kalman Gains
# # =====================================

# kalman_gains = np.array(kalman_gains)

# plt.figure(figsize=figure_size)
# k_x     = kalman_gains[:, 0]
# k_y     = kalman_gains[:, 1]
# k_v     = kalman_gains[:, 2]
# k_theta = kalman_gains[:, 3]

# plt.title("Kalman gains by state")
# plt.xlabel("Time Step")
# plt.ylabel("Kalman Gain Value")

# plt.plot(t, k_x, label="K(x)")
# plt.plot(t, k_y, label="K(y)")
# plt.plot(t, k_v, label="K(v)")
# plt.plot(t, k_theta, label="K(theta)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Extract Kalman Gains for each state

# # =====================================
# # Debug Figure: x_cov and heading
# # =====================================

# plt.figure(figsize=figure_size)
# plt.title("x_cov and heading plot")
# plt.xlabel("Time Step")
# plt.ylabel("Value")

# var_x = covariances[:, 0, 0]
# var_v = covariances[:, 2, 2]

# v_s = x_est[:, 2]
# v_s_true = x_traj[:, 2]
# thetas = x_est[:, 3]

# jacob_elem = -(-v_s*np.sin(thetas))

# plt.plot(t, var_x, label = "Var(x)")
# plt.plot(t, var_v, label = "Var(v)")

# plt.plot(t, x_est[:, 3], label= "theta")
# plt.plot(t, v_s, label= "v_est")
# plt.plot(t, v_s_true, label= "v_true")
# plt.plot(t, jacob_elem, label="-(-v*sin(theta))")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# =====================================
# Debug Figure: x_cov, cov_v, cov_theta
# =====================================

# fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

# var_x = covariances[:, 0, 0]
# var_y = covariances[:, 1, 1]
# var_v = covariances[:, 2, 2]
# var_theta = covariances[:, 3, 3]

# # x covariance
# axs[0].plot(t, var_x, label=r'$\Sigma_{xx}$')
# axs[0].set_ylabel(r'Covariance')
# axs[0].legend()
# axs[0].grid(True)

# # y, v, theta covariances
# axs[1].plot(t, var_y, linewidth=5.0, label=r'$\Sigma_{yy}$')
# axs[1].plot(t, var_v, label=r'$\Sigma_{vv}$')
# axs[1].plot(t, var_theta, "--", color='white', label=r'$\Sigma_{\theta\theta}$')

# axs[1].set_ylabel(r'Covariance')
# axs[1].set_xlabel(r'Time step')
# axs[1].legend()
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()

var_x = covariances[:, 0, 0]
var_y = covariances[:, 1, 1]
var_v = covariances[:, 2, 2]
var_theta = covariances[:, 3, 3]
trace_cov = np.trace(covariances, axis1=1, axis2=2)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(t, var_x, label=r'$\Sigma_{xx}$')
ax.plot(t, var_y, linewidth=5.0, label=r'$\Sigma_{yy}$')
ax.plot(t, var_v, label=r'$\Sigma_{vv}$')
ax.plot(t, var_theta, "k-", label=r'$\Sigma_{\theta\theta}$')  # make theta thicker
ax.plot(t, trace_cov, label="Trace($\Sigma$)")

# # Inset axes
# # axins = inset_axes(ax, width="40%", height="40%", loc='upper right')  # position inset relative to parent
# axins = fig.add_axes([0.55, 0.55, 0.35, 0.35])  # adjust as needed
# axins.plot(t, var_x)
# axins.plot(t, var_y, linewidth=5.0)
# axins.plot(t, var_v)
# axins.plot(t, var_theta, "k-")
# axins.plot(t, trace_cov)

# # Set inset limits
# axins.set_xlim(-0.05, 6.0)
# axins.set_ylim(-0.05, 0.11)
# axins.grid(True)

# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")  # loc1, loc2 = corners to connect

text_size = 14
ax.set_xlabel(r'Time (s)', fontsize=text_size)
ax.set_ylabel(r'Covariance', fontsize=text_size)
ax.legend(fontsize=text_size)
ax.tick_params(labelsize=text_size)
ax.grid(True)
fig.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.14)

plt.show()
