import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from enum import Enum

class Color(Enum):
    EKF_TRUE = "blue"
    EKF_MEAS = "blue"
    EKF_EST = "cyan"
    
    GEKF_TRUE = "red"
    GEKF_MEAS = "red"
    GEKF_EST = "purple"

case = "sinusoidal"

FONTSIZE = 15
LINEWIDTH = 3.0

def plot_ekf_gekf(
    ekf_path: str = "single_int_EKF.npz",
    gekf_path: str = "single_int_GEKF.npz",
    setpoint: float = 6.0,
    safety_boundary: float = 5.0,
    zoom_xlim: tuple = (4.0, 5.1),
    zoom_ylim: tuple = (4.95, 5.01),
    save_path: str = None,
    show_inset_separate: bool = False,
):
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

    def _draw_inset(ax, linewidth_ekf=1.5, linewidth_gekf=3.0):
        """Draw inset content onto any axes object."""
        ax.axhline(safety_boundary, color="red",     linestyle="--", linewidth=2.0)
        ax.axhline(setpoint,        color="magenta", linestyle="--", linewidth=2.0)

        ax.plot(ekf_time, ekf_true, color=Color.EKF_TRUE.value,  linewidth=linewidth_ekf)
        ax.scatter(ekf_time, ekf_meas, color=Color.EKF_MEAS.value,  marker="o", s=10, alpha=0.01)
        ax.plot(ekf_time, ekf_est,  color=Color.EKF_EST.value,  linewidth=linewidth_ekf, linestyle="--", dashes=(6, 3))

        ax.plot(gekf_time, gekf_true, color=Color.GEKF_TRUE.value, linewidth=1.5, linestyle="-")
        ax.scatter(gekf_time, gekf_meas, color=Color.GEKF_MEAS.value, marker="x", s=14, alpha=0.01)
        ax.plot(gekf_time, gekf_est,  color=Color.GEKF_EST.value, linewidth=linewidth_gekf, linestyle="--", dashes=(6, 3), markevery=50, markersize=4)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(*zoom_xlim)
        ax.set_ylim(*zoom_ylim)
        ax.grid(True, linestyle="--", alpha=0.3)

    # ── figure & main axes ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.axhline(safety_boundary, color="red",    linestyle="--", linewidth=2.0)
    ax.axhline(setpoint,        color="purple", linestyle="--", linewidth=2.0)

    ax.scatter(ekf_time, ekf_meas, color=Color.EKF_MEAS.value,  marker="o", s=12, alpha=0.01, label="EKF - Measured", zorder=2)
    ax.scatter(gekf_time, gekf_meas, color=Color.GEKF_MEAS.value, marker="x", s=18, alpha=0.01, label="GEKF - Measured", zorder=2)
    
    ax.plot(ekf_time, ekf_est,  color=Color.EKF_EST.value,  linewidth=1.8, linestyle="--", dashes=(6, 3), label="EKF - Estimated")
    ax.plot(gekf_time, gekf_est,  color=Color.GEKF_EST.value, linewidth=3.5, linestyle="--", dashes=(6, 3), label="GEKF - Estimated", marker="s", markevery=30, markersize=4)
    
    ax.plot(gekf_time, gekf_true, color=Color.GEKF_TRUE.value, linewidth=1.5, linestyle="-", label="GEKF - True")
    ax.plot(ekf_time, ekf_true, color=Color.EKF_TRUE.value,  linewidth=1.8, label="EKF - True")

    ax.tick_params(labelsize=FONTSIZE)
    ax.set_xlabel("Time (s)", fontsize=FONTSIZE)
    ax.set_ylabel("Position",  fontsize=FONTSIZE)
    ax.grid(True, linestyle="-")
    legend = ax.legend(loc="lower right", fontsize=FONTSIZE)
    
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    # ── inset ─────────────────────────────────────────────────────────────
    if not show_inset_separate:
        axins = ax.inset_axes([0.02, 0.50, 0.38, 0.3])
        _draw_inset(axins)
        axins.tick_params(labelsize=7)
        axins.set_xticks([])
        axins.set_yticks([])
        ax.indicate_inset_zoom(axins, edgecolor="blue", linewidth=2.0)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}/single_int_comparison.png", dpi=600, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

    # ── separate inset figure ──────────────────────────────────────────────
    if show_inset_separate:
        fig_inset, ax_inset = plt.subplots(figsize=(5, 4))
        _draw_inset(ax_inset, linewidth_ekf=1.5, linewidth_gekf=10.0)

        if save_path:
            inset_save = f"{save_path}/inset.png"
            fig_inset.savefig(inset_save, dpi=300, bbox_inches="tight")
            print(f"Inset saved to {inset_save}")

        plt.tight_layout()
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
        save_path="./",
        show_inset_separate = True
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
    main_ax.plot(x_traj_ekf[:, 0], x_traj_ekf[:, 1],      color=Color.EKF_TRUE.value, linestyle="-", alpha=0.8, label="EKF - True")
    main_ax.plot(x_est_ekf[:, 0], x_est_ekf[:, 1],        color=Color.EKF_EST.value, linestyle="--", label="EKF - Estimated")
    main_ax.scatter(x_meas_ekf[:, 0], x_meas_ekf[:, 1],   color=Color.EKF_MEAS.value, marker="o", s=6, alpha=0.01)  # no label

    # GEKF
    main_ax.plot(x_traj_gekf[:, 0], x_traj_gekf[:, 1],    color=Color.GEKF_TRUE.value, linestyle="-", alpha=0.8, label="GEKF - True")
    main_ax.plot(x_est_gekf[:, 0], x_est_gekf[:, 1],      color=Color.GEKF_EST.value, linestyle="--", label="GEKF - Estimated")
    main_ax.scatter(x_meas_gekf[:, 0], x_meas_gekf[:, 1], color=Color.GEKF_MEAS.value, marker="x", s=8, alpha=0.01)  # no label

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
    inset_ax.plot(x_traj_ekf[:, 0], x_traj_ekf[:, 1], color=Color.EKF_TRUE.value, linestyle="-", alpha=0.8, label="True x (EKF)", linewidth=5)
    inset_ax.plot(x_est_ekf[:, 0], x_est_ekf[:, 1], color=Color.EKF_EST.value, linestyle="--", label="Estimated x (EKF)", linewidth=2.5)
    inset_ax.scatter(x_meas_ekf[:, 0], x_meas_ekf[:, 1], color=Color.EKF_MEAS.value, marker="o", s=4, alpha=0.01)

    # GEKF inset
    inset_ax.plot(x_traj_gekf[:, 0], x_traj_gekf[:, 1], color=Color.GEKF_TRUE.value, linestyle="-", alpha=0.8, label="True x (GEKF)", linewidth=5)
    inset_ax.plot(x_est_gekf[:, 0], x_est_gekf[:, 1], color=Color.GEKF_EST.value, linestyle="--", label="Estimated x (GEKF)", linewidth=2.5)
    inset_ax.scatter(x_meas_gekf[:, 0], x_meas_gekf[:, 1], color=Color.GEKF_MEAS.value, marker="x", s=5, alpha=0.01)

    inset_ax.axhline(y=wall_y, color="red", linestyle="dashed", linewidth=5)
    inset_ax.plot(x_nom_ekf[:, 0], x_nom_ekf[:, 1], color="black", linestyle="-", linewidth=2.5, label="Nominal trajectory")

    # main_ax.indicate_inset_zoom(inset_ax, edgecolor="blue")
    mark_inset(main_ax, inset_ax,
            loc1=2, loc2=3,   # corners to connect
            fc="none", ec="blue")

    # plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.999, bottom=0.08, top=0.999)
    plt.savefig("Trajectories_comparison_with_inset.png", dpi=600)

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