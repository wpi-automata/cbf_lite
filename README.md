# cbf_lite: Overview and Setup

This repo implements CBF (Control Barrier Function) QPs (Quadtratic Programs) for setpoint and trajectory tracking scenarios. The code in this repo, especially for the sensor and estimator models, was adapted from https://github.com/bardhh/cbfkit. This repo specifically implements the GEKF (https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-spr.2013.0161) and the Belief CBF (https://arxiv.org/abs/2309.06499). Please view our corresponding paper: https://arxiv.org/abs/2510.14100. 

## Simulation Scripts Overview

| Script | Nominal Controller | CBF Type | Dynamics | Sensor | Estimator | Control Objective | Optimization Package |
|------|-------------------|----------|----------|--------|-----------|-------------------|----------------------|
| `simple_dynamics_vanilla_cbf.py` | CLF | 1× Vanilla | 2D Linear Double Integrator | Standard | None | Obstacle Avoidance | cvxpy |
| `sim_vanilla_cbf_single_integrator1D.py` | CLF | 1× Vanilla | 1D Non-linear Single Integrator | Standard / Multiplicative | EKF / GEKF | Setpoint tracking | jaxopt |
| `sim_vanilla_cbf_wall.py` | CLF | 1× Vanilla | 2D Linear Double Integrator | Standard | EKF | Setpoint tracking in y-dim | jaxopt |
| `sim_belief_cbf_single_integrator1D.py`<br>`sim_belief_cbf_single_integrator1D_batch.py` | CLF | 1× Belief | 1D Single Integrator | Standard / Multiplicative | EKF / GEKF | Setpoint tracking | jaxopt |
| `sim_belief_cbf_double_integrator_1D.py`<br>`sim_belief_cbf_double_integrator_1D_batch.py` | CLF | 1× Belief (2nd Order) | 1D Double Integrator | Standard / Multiplicative | EKF / GEKF | Setpoint tracking | jaxopt |
| `sim_belief_cbf_wall.py` | CLF | 1× Belief | 1D Non-linear Single Integrator | Standard / Multiplicative | EKF / GEKF | Setpoint tracking in y-dim | jaxopt |
| `sim_belief_cbf_2D_dubins.py` | CLF | 2× Belief (2nd Order) | 2D Dubins | Standard / Multiplicative | EKF / GEKF | Trajectory tracking | jaxopt |
| `sim_belief_cbf_2D_dubins_sinusoidal.py` | CLF | 2× Belief (2nd Order) | 2D Dubins | Standard / Multiplicative | EKF / GEKF | Trajectory tracking | jaxopt |
| `sim_belief_cbf_2D_dubins_sinusoidal_gain_scheduling.py`<br>`sim_belief_cbf_2D_dubins_sinusoidal_gain_scheduling_batch.py` | Gain Scheduled Controller | 2× Belief (2nd Order) | 2D Dubins | Standard / Multiplicative | EKF / GEKF | Trajectory tracking | jaxopt |
| `sim_mixed_dynamics.py` | CLF | 2× Belief | 1D Non-linear Single Integrator (Prediction) / 2D Dubins (Actuation) | Standard / Multiplicative | EKF / GEKF | Setpoint tracking in y-dim | jaxopt |
| `sim_mixed_dynamics_double_int.py` | CLF | 2× Belief (2nd Order) | 2D Double Integrator (Prediction) / 2D Dubins (Actuation) | Standard / Multiplicative | EKF / GEKF | Setpoint tracking in y-dim | jaxopt |


## Setup
### Create the Environment

To get started, it's is recommended to create a conda environment.

```bash
conda create -n cbf_lite python=3.12 -y
conda activate cbf_lite
```

### Install Dependencies

This repo mainly depends on `jaxopt` and `jax` for implementing QPs. Please install the following packages in your conda environment to run all the scripts:

```bash
pip install \
    numpy \
    matplotlib \
    cvxpy \
    jax \
    jaxopt \
    tqdm
```

### Verify Installation

```bash
python -c "import numpy, matplotlib, cvxpy, jax, jaxopt, tqdm; print('cbf_lite environment ready')"
```
