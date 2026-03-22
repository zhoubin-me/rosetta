#!/usr/bin/env python3
# Copyright 2026 Brian Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Closed-loop smoother for RobotEra Q5 HybridJointCommand policy output."""

from __future__ import annotations

from collections import deque
import copy
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xbot_common_interfaces.msg import HybridJointCommand

from .q5_action_smoother import compute_position_step, ramp_gains


@dataclass(slots=True)
class TopicRuntimeState:
    published: int = 0
    reached: int = 0
    timeouts: int = 0
    dropped_targets: int = 0


class Q5ActionSmootherNode(Node):
    """Interpolate live Q5 arm commands toward policy targets."""

    def __init__(self) -> None:
        super().__init__("q5_action_smoother")

        self.declare_parameter("input_topic", "/hil/policy_raw/wr1_controller/commands")
        self.declare_parameter("output_topic", "/hil/policy/wr1_controller/commands")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("wait_for_joint_state_sec", 2.0)
        self.declare_parameter("control_period_sec", 0.01)
        self.declare_parameter("joint_tolerance", 0.02)
        self.declare_parameter("max_step_per_cycle", 0.03)
        self.declare_parameter("reach_timeout_sec", 5.0)
        self.declare_parameter("kp_start_scale", 0.5)
        self.declare_parameter("kd_start_scale", 0.5)
        self.declare_parameter("gain_ramp_sec", 1.0)
        self.declare_parameter("drop_intermediate_targets", True)

        self._input_topic = str(self.get_parameter("input_topic").value)
        self._output_topic = str(self.get_parameter("output_topic").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._wait_for_joint_state_sec = float(self.get_parameter("wait_for_joint_state_sec").value)
        self._control_period_sec = float(self.get_parameter("control_period_sec").value)
        self._joint_tolerance = float(self.get_parameter("joint_tolerance").value)
        self._max_step_per_cycle = float(self.get_parameter("max_step_per_cycle").value)
        self._reach_timeout_sec = float(self.get_parameter("reach_timeout_sec").value)
        self._kp_start_scale = float(self.get_parameter("kp_start_scale").value)
        self._kd_start_scale = float(self.get_parameter("kd_start_scale").value)
        self._gain_ramp_sec = float(self.get_parameter("gain_ramp_sec").value)
        self._drop_intermediate_targets = bool(
            self.get_parameter("drop_intermediate_targets").value
        )

        if self._control_period_sec <= 0.0:
            raise RuntimeError("control_period_sec must be > 0")
        if self._joint_tolerance < 0.0:
            raise RuntimeError("joint_tolerance must be >= 0")
        if self._max_step_per_cycle <= 0.0:
            raise RuntimeError("max_step_per_cycle must be > 0")
        if self._kp_start_scale < 0.0 or self._kd_start_scale < 0.0:
            raise RuntimeError("kp_start_scale and kd_start_scale must be >= 0")

        self._latest_joint_positions: dict[str, float] = {}
        self._seen_joint_state = False
        self._joint_state_warning_logged = False
        self._startup_time = time.monotonic()

        self._active_target: HybridJointCommand | None = None
        self._pending_target: HybridJointCommand | None = None
        self._pending_queue: deque[HybridJointCommand] = deque()
        self._active_target_start_time = 0.0
        self._state = TopicRuntimeState()

        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._joint_state_cb,
            100,
        )
        self.create_subscription(
            HybridJointCommand,
            self._input_topic,
            self._input_cb,
            50,
        )
        self._publisher = self.create_publisher(HybridJointCommand, self._output_topic, 50)
        self._timer = self.create_timer(self._control_period_sec, self._control_cycle)

        self.get_logger().info(
            f"Q5 smoother input={self._input_topic}, output={self._output_topic}, "
            f"joint_state_topic={self._joint_state_topic}, period={self._control_period_sec:.3f}s, "
            f"tol={self._joint_tolerance:.4f}, max_step={self._max_step_per_cycle:.4f}, "
            f"timeout={self._reach_timeout_sec:.2f}s, drop_intermediate_targets="
            f"{self._drop_intermediate_targets}"
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self._latest_joint_positions[name] = position
        self._seen_joint_state = True

    def _input_cb(self, msg: HybridJointCommand) -> None:
        target = copy.deepcopy(msg)

        if self._active_target is None:
            self._set_active_target(target)
            return

        if self._drop_intermediate_targets:
            if self._pending_target is not None:
                self._state.dropped_targets += 1
            self._pending_target = target
            self._pending_queue.clear()
            return

        self._pending_queue.append(target)

    def _set_active_target(self, msg: HybridJointCommand) -> None:
        self._active_target = msg
        self._active_target_start_time = time.monotonic()

    def _promote_pending_target(self) -> None:
        next_target: HybridJointCommand | None = None
        if self._pending_target is not None:
            next_target = self._pending_target
            self._pending_target = None
        elif self._pending_queue:
            next_target = self._pending_queue.popleft()

        if next_target is None:
            self._active_target = None
            return

        self._active_target = next_target
        self._active_target_start_time = time.monotonic()

    def _control_cycle(self) -> None:
        if not self._seen_joint_state and not self._joint_state_warning_logged:
            since_start = time.monotonic() - self._startup_time
            if since_start >= self._wait_for_joint_state_sec:
                self.get_logger().warn(
                    f"No {self._joint_state_topic} received in "
                    f"{self._wait_for_joint_state_sec:.2f}s; missing joints fall back to target."
                )
                self._joint_state_warning_logged = True

        if self._active_target is None:
            if self._pending_target is not None or self._pending_queue:
                self._promote_pending_target()
            else:
                return

        assert self._active_target is not None
        out_msg, reached = self._build_interpolated_step(self._active_target)
        self._publish(out_msg)

        elapsed_sec = time.monotonic() - self._active_target_start_time
        if reached:
            self._publish_exact_target(self._active_target, elapsed_sec)
            self._state.reached += 1
            self._promote_pending_target()
            return

        if self._reach_timeout_sec > 0.0 and elapsed_sec >= self._reach_timeout_sec:
            self.get_logger().warn(
                f"Timeout waiting for target on {self._output_topic} after {elapsed_sec:.2f}s; "
                "advancing to latest queued target."
            )
            self._publish_exact_target(self._active_target, elapsed_sec)
            self._state.timeouts += 1
            self._promote_pending_target()

    def _build_interpolated_step(
        self,
        msg: HybridJointCommand,
    ) -> tuple[HybridJointCommand, bool]:
        joint_names = list(msg.joint_name)
        current_positions = [
            self._latest_joint_positions.get(joint_name, target_position)
            for joint_name, target_position in zip(joint_names, msg.position)
        ]
        next_positions, reached = compute_position_step(
            current=current_positions,
            target=list(msg.position),
            joint_tolerance=self._joint_tolerance,
            max_step_per_cycle=self._max_step_per_cycle,
        )

        out = self._copy_command(msg)
        out.position = next_positions
        self._apply_gain_ramp(out)
        return out, reached

    def _publish_exact_target(self, msg: HybridJointCommand, elapsed_sec: float) -> None:
        out = self._copy_command(msg)
        out.position = list(msg.position)
        self._apply_gain_ramp(out, elapsed_sec)
        self._publish(out)

    def _copy_command(self, msg: HybridJointCommand) -> HybridJointCommand:
        out = HybridJointCommand()
        out.header = copy.deepcopy(msg.header)
        out.joint_name = list(msg.joint_name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.kp = list(msg.kp)
        out.kd = list(msg.kd)

        if hasattr(msg, "feedforward") and hasattr(out, "feedforward"):
            out.feedforward = list(msg.feedforward)
        elif hasattr(msg, "effort") and hasattr(out, "effort"):
            out.effort = list(msg.effort)

        return out

    def _apply_gain_ramp(
        self,
        msg: HybridJointCommand,
        elapsed_sec: float | None = None,
    ) -> None:
        if elapsed_sec is None:
            elapsed_sec = time.monotonic() - self._active_target_start_time
        msg.kp, msg.kd = ramp_gains(
            kp=list(msg.kp),
            kd=list(msg.kd),
            elapsed_sec=elapsed_sec,
            kp_start_scale=self._kp_start_scale,
            kd_start_scale=self._kd_start_scale,
            gain_ramp_sec=self._gain_ramp_sec,
        )

    def _publish(self, msg: HybridJointCommand) -> None:
        self._publisher.publish(msg)
        self._state.published += 1


def main(args=None) -> int:
    rclpy.init(args=args)
    node = None
    try:
        node = Q5ActionSmootherNode()
        rclpy.spin(node)
        return 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
