"""Custom encoders for RobotEra Q5 contracts."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from rosidl_runtime_py.utilities import get_message

from .common.contract import ActionStreamSpec


ARM_KP_DEFAULT = 300.0
ARM_KD_DEFAULT = 50.0


def encode_q5_arm_hybrid_joint_command(
    action_vec: Sequence[float],
    spec: ActionStreamSpec,
    stamp_ns: int | None = None,
):
    """Encode arm-only position actions into HybridJointCommand.

    Contract selector names are expected to be:
      - position.<joint_name>

    This encoder fills:
      - velocity = 0
      - feedforward = 0
      - kp = 300
      - kd = 50
    for each commanded arm joint.
    """
    msg_cls = get_message("xbot_common_interfaces/msg/HybridJointCommand")
    msg = msg_cls()

    if stamp_ns is not None:
        try:
            msg.header.stamp.sec = stamp_ns // 1_000_000_000
            msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        except AttributeError:
            pass

    arr = np.asarray(action_vec, dtype=np.float64).flatten()
    names = list(spec.names or [])

    if len(names) != len(arr):
        raise ValueError(f"names length ({len(names)}) != action length ({len(arr)})")

    joint_names: list[str] = []
    positions: list[float] = []
    for i, selector in enumerate(names):
        if "." in selector:
            field, joint_name = selector.split(".", 1)
        else:
            field, joint_name = "position", selector

        if field != "position":
            raise ValueError(
                f"Unsupported selector field '{field}' for Q5 arm encoder. "
                "Use only 'position.<joint_name>' selectors."
            )

        joint_names.append(joint_name)
        positions.append(float(arr[i]))

    n = len(joint_names)
    msg.joint_name = joint_names
    msg.position = positions
    msg.velocity = [0.0] * n
    msg.feedforward = [0.0] * n
    msg.kp = [ARM_KP_DEFAULT] * n
    msg.kd = [ARM_KD_DEFAULT] * n
    return msg
