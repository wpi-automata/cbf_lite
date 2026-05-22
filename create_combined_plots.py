import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


case = "single_int" # "sinusoidal"


def plot_ekf_gekf(
    ekf_path: str = "single_int_EKF.npz",
    gekf_path: str = "single_int_GEKF.npz",
    setpoint: float = 6.0,
    safety_boundary: float = 5.0,
    zoom_xlim: tuple = (3.5, 5.1),
    zoom_ylim: tuple = (4.9, 5.1),
    save_path: str = None,
):
    """
    Recreate the EKF vs GEKF trajectory comparison plot.

    Parameters
    ----------
    ekf_path        : path to EKF .npz file
    gekf_path       : path to GEKF .npz file
    setpoint        : dashed magenta reference line
    safety_boundary : dashed red reference line
    zoom_xlim/ylim  : region shown in the inset zoom box
    save_path       : if given, saves the figure to this path
    """
    # ── load data ──────────────────────────────────────────────────────────
    ekf  = np.load(ekf_path)
    gekf = np.load(gekf_path)

    ekf_time  = ekf["time"]
    ekf_true  = ekf["x_traj"]
    ekf_meas  = ekf["x_meas"]
    ekf_est   = ekf["x_est"]

    gekf_time = gekf["time"]
    gekf_true = gekf["x_traj"]
    gekf_meas = gekf["x_meas"]
    gekf_est  = gekf["x_est"]

    # ── figure & main axes ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    # reference lines
    ax.axhline(safety_boundary, color="red",     linestyle="--", linewidth=2.0)
        #        label="Safety Boundary")
    ax.axhline(setpoint,        color="purple", linestyle="--", linewidth=2.0)
        #        label="Setpoint")

    # EKF
    ax.scatter(ekf_time, ekf_meas, color="cyan",  marker="o", s=12, alpha=0.2,
               label="EKF - Measured", zorder=2)
    ax.plot(ekf_time, ekf_true, color="blue",  linewidth=1.8,
            label="EKF - True")
    ax.plot(ekf_time, ekf_est,  color="blue",  linewidth=1.8, linestyle="--",
            dashes=(6, 3), label="EKF - Estimated")

    # GEKF
    ax.scatter(gekf_time, gekf_meas, color="salmon", marker="x", s=18, alpha=0.2,
               label="GEKF - Measured", zorder=2)
    ax.plot(gekf_time, gekf_est,  color="orange",  linewidth=3.5, linestyle="--"
            , dashes=(4, 4), label="GEKF - Estimated",
            marker="s", markevery=30, markersize=4)
    ax.plot(gekf_time, gekf_true, color="black", linewidth=1.5,
            linestyle="-", label="GEKF - True")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Position",  fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="-")

    # ── inset zoom ─────────────────────────────────────────────────────────
    axins = ax.inset_axes([0.02, 0.50, 0.38, 0.3])   # [x0, y0, w, h] in axes fraction

    # reference lines in inset
    axins.axhline(safety_boundary, color="red",     linestyle="--", linewidth=1.0)
    axins.axhline(setpoint,        color="magenta", linestyle="--", linewidth=1.0)

    # EKF in inset
    axins.scatter(ekf_time, ekf_meas, color="cyan",  marker="o", s=10, alpha=0.1)
    axins.plot(ekf_time, ekf_true, color="blue",  linewidth=1.5)
    axins.plot(ekf_time, ekf_est,  color="blue",  linewidth=1.5, linestyle="--",
               dashes=(6, 3))

    # GEKF in inset
    axins.scatter(gekf_time, gekf_meas, color="salmon", marker="x", s=14, alpha=0.1)
    axins.plot(gekf_time, gekf_est,  color="orange",  linewidth=6.0,
               linestyle="--", dashes=(6, 3), markevery=50, markersize=4)
    axins.plot(gekf_time, gekf_true, color="black", linewidth=1.5, linestyle="-")

    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.tick_params(labelsize=7)
    axins.grid(True, linestyle="--", alpha=0.3)
    axins.set_xticks([])
    axins.set_yticks([])

    # draw the zoom box and connector lines
    ax.indicate_inset_zoom(axins, edgecolor="blue", linewidth=2.0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

if case == "single_int":
    parser = argparse.ArgumentParser(description="Plot EKF vs GEKF trajectories")
    parser.add_argument("--ekf",      default="single_int_EKF.npz",  help="Path to EKF .npz")
    parser.add_argument("--gekf",     default="single_int_GEKF.npz", help="Path to GEKF .npz")
    parser.add_argument("--setpoint", default=6.0,  type=float)
    parser.add_argument("--safety",   default=5.0,  type=float)
    parser.add_argument("--save",     default=None, help="Save figure to this path")
    args = parser.parse_args()

    plot_ekf_gekf(
        ekf_path=args.ekf,
        gekf_path=args.gekf,
        setpoint=args.setpoint,
        safety_boundary=args.safety,
        save_path=args.save,
    )

elif case == "sinusoidal":
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