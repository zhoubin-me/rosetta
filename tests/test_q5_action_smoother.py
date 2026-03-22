import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "rosetta" / "q5_action_smoother.py"
SPEC = importlib.util.spec_from_file_location("q5_action_smoother", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

compute_position_step = MODULE.compute_position_step
ramp_gains = MODULE.ramp_gains


def test_compute_position_step_clamps_each_joint():
    next_position, reached = compute_position_step(
        current=[0.0, 0.9, -1.0],
        target=[0.2, 1.0, -0.8],
        joint_tolerance=0.05,
        max_step_per_cycle=0.1,
    )

    assert next_position == [0.1, 1.0, -0.9]
    assert reached is False


def test_compute_position_step_marks_reached_when_all_joints_in_tolerance():
    next_position, reached = compute_position_step(
        current=[1.0, -1.0],
        target=[1.01, -1.02],
        joint_tolerance=0.05,
        max_step_per_cycle=0.1,
    )

    assert next_position == [1.01, -1.02]
    assert reached is True


def test_ramp_gains_scales_from_start_to_full_gain():
    kp, kd = ramp_gains(
        kp=[300.0, 100.0],
        kd=[50.0, 20.0],
        elapsed_sec=0.5,
        kp_start_scale=0.5,
        kd_start_scale=0.2,
        gain_ramp_sec=1.0,
    )

    assert kp == [225.0, 75.0]
    assert kd == pytest.approx([30.0, 12.0])


def test_ramp_gains_returns_original_values_when_ramp_disabled():
    kp, kd = ramp_gains(
        kp=[1.0],
        kd=[2.0],
        elapsed_sec=10.0,
        kp_start_scale=0.1,
        kd_start_scale=0.1,
        gain_ramp_sec=0.0,
    )

    assert kp == [1.0]
    assert kd == [2.0]
