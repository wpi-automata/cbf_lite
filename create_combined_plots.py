import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Load saved runs
ekf_data = np.load("sim_sinusoidal_EKF_v_5.0.npz")
gekf_data = np.load("sim_sinusoidal_GEKF_v_5.0.npz")

# Extract variables
x_traj_ekf = ekf_data["x_traj"]
x_est_ekf  = ekf_data["x_est"]
x_meas_ekf = ekf_data["x_meas"]
covariances_ekf = ekf_data["covariances"]
cov_trace_ekf = np.trace(covariances_ekf, axis1=1, axis2=2)
x_nom_ekf  = ekf_data["x_nom"]

x_traj_gekf = gekf_data["x_traj"]
x_est_gekf  = gekf_data["x_est"]
x_meas_gekf = gekf_data["x_meas"]
covariances_gekf = gekf_data["covariances"]
cov_trace_gekf = np.trace(covariances_gekf, axis1=1, axis2=2)
x_nom_gekf  = gekf_data["x_nom"]

time_ekf = ekf_data["time"]
time_gekf = gekf_data["time"]

wall_y = 5.0

# --- Plot trajectory comparison (main 2D plot) ---
fig, main_ax = plt.subplots(figsize=(15, 15))

# EKF
main_ax.plot(x_traj_ekf[:, 0], x_traj_ekf[:, 1], color="blue", linestyle="-", alpha=0.8, label="EKF - True")
main_ax.plot(x_est_ekf[:, 0], x_est_ekf[:, 1], color="purple", linestyle="--", label="EKF - Estimated")
main_ax.scatter(x_meas_ekf[:, 0], x_meas_ekf[:, 1],
                color="skyblue", marker="o", s=6, alpha=0.15)  # no label

# GEKF
main_ax.plot(x_traj_gekf[:, 0], x_traj_gekf[:, 1], color="red", linestyle="-", alpha=0.8, label="GEKF - True")
main_ax.plot(x_est_gekf[:, 0], x_est_gekf[:, 1], color="green", linestyle="--", label="GEKF - Estimated")
main_ax.scatter(x_meas_gekf[:, 0], x_meas_gekf[:, 1],
                color="salmon", marker="x", s=8, alpha=0.15)  # no label

# Boundaries
main_ax.axhline(y=wall_y, color="red", linestyle="dashed", linewidth=1, label="Safety Boundary")
main_ax.axhline(y=-wall_y, color="red", linestyle="dashed", linewidth=1)

# Nominal trajectory
main_ax.plot(x_nom_ekf[:, 0], x_nom_ekf[:, 1], color="black", linestyle="-", linewidth=2.5, label="Nominal trajectory")

text_size = 22

main_ax.set_xlabel("x [m]", fontsize=text_size)
main_ax.set_ylabel("y [m]", fontsize=text_size)
main_ax.tick_params(axis="both", which="major", labelsize=text_size)
main_ax.grid(True)

# --- Custom opaque handles for measurements ---
meas_ekf_handle = mlines.Line2D([], [], color="skyblue", marker="o", linestyle="None",
                                markersize=8, alpha=1.0, label="Measurements (EKF)")
meas_gekf_handle = mlines.Line2D([], [], color="salmon", marker="x", linestyle="None",
                                 markersize=8, alpha=1.0, label="Measurements (GEKF)")

# Collect existing handles + add custom measurement handles
handles, labels = main_ax.get_legend_handles_labels()
handles.extend([meas_ekf_handle, meas_gekf_handle])
labels.extend(["Measurements (EKF)", "Measurements (GEKF)"])
main_ax.legend(handles, labels, loc="lower left", fontsize=text_size)

# --- Inset: X-component comparison over time ---
inset_ax = main_ax.inset_axes([0.4, 0.625, 0.35, 0.35],  # [x0, y0, width, height] in figure coords
                              xlim=[10, 50], ylim=[3.75, 5.25],
                              xticklabels=[], yticklabels=[])

# EKF inset
inset_ax.plot(x_traj_ekf[:, 0], x_traj_ekf[:, 1], color="blue", linestyle="-", alpha=0.8, label="True x (EKF)", linewidth=5)
inset_ax.plot(x_est_ekf[:, 0], x_est_ekf[:, 1], color="purple", linestyle="--", label="Estimated x (EKF)", linewidth=2.5)
inset_ax.scatter(x_meas_ekf[:, 0], x_meas_ekf[:, 1], color="skyblue", marker="o", s=4, alpha=0.3)

# GEKF inset
inset_ax.plot(x_traj_gekf[:, 0], x_traj_gekf[:, 1], color="red", linestyle="-", alpha=0.8, label="True x (GEKF)", linewidth=5)
inset_ax.plot(x_est_gekf[:, 0], x_est_gekf[:, 1], color="green", linestyle="--", label="Estimated x (GEKF)", linewidth=2.5)
inset_ax.scatter(x_meas_gekf[:, 0], x_meas_gekf[:, 1], color="salmon", marker="x", s=5, alpha=0.3)

inset_ax.axhline(y=wall_y, color="red", linestyle="dashed", linewidth=5)
inset_ax.plot(x_nom_ekf[:, 0], x_nom_ekf[:, 1], color="black", linestyle="-", linewidth=2.5, label="Nominal trajectory")

# main_ax.indicate_inset_zoom(inset_ax, edgecolor="blue")
mark_inset(main_ax, inset_ax,
           loc1=2, loc2=3,   # corners to connect
           fc="none", ec="blue")

# plt.tight_layout()
fig.subplots_adjust(left=0.08, right=0.999, bottom=0.08, top=0.999)
plt.savefig("Trajectories_comparison_with_inset.png", dpi=300)

# # --- Plot covariance trace comparison ---
# plt.figure(figsize=(8, 8))
# plt.plot(time_ekf, cov_trace_ekf, "b-", label="EKF")
# plt.plot(time_gekf, cov_trace_gekf, "g-", label="GEKF")
# plt.title("Trace of Covariance: EKF vs GEKF")
# plt.xlabel("Time [s]", fontsize=14)
# plt.ylabel("Trace(P)", fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.savefig("CovarianceTrace_comparison.png", dpi=300)

plt.show()