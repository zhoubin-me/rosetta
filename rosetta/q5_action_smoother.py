"""Pure helpers for Q5 action smoothing."""

from __future__ import annotations


def compute_position_step(
    current: list[float],
    target: list[float],
    joint_tolerance: float,
    max_step_per_cycle: float,
) -> tuple[list[float], bool]:
    """Clamp per-joint motion toward target and report whether target is reached."""
    if len(current) != len(target):
        raise ValueError(
            f"current length ({len(current)}) does not match target length ({len(target)})"
        )

    next_position: list[float] = []
    reached = True

    for current_value, target_value in zip(current, target):
        error = target_value - current_value
        if abs(error) > joint_tolerance:
            reached = False

        step = max(-max_step_per_cycle, min(max_step_per_cycle, error))
        next_position.append(current_value + step)

    return next_position, reached


def ramp_gains(
    kp: list[float],
    kd: list[float],
    elapsed_sec: float,
    kp_start_scale: float,
    kd_start_scale: float,
    gain_ramp_sec: float,
) -> tuple[list[float], list[float]]:
    """Scale gains from a reduced startup value toward 1.0."""
    if gain_ramp_sec <= 0.0:
        return list(kp), list(kd)

    alpha = max(0.0, min(1.0, elapsed_sec / gain_ramp_sec))
    kp_scale = kp_start_scale + (1.0 - kp_start_scale) * alpha
    kd_scale = kd_start_scale + (1.0 - kd_start_scale) * alpha
    return [value * kp_scale for value in kp], [value * kd_scale for value in kd]
