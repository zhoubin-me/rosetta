#!/usr/bin/env python3

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import JointState
from xbot_common_interfaces.msg import HybridJointCommand


@dataclass
class TopicRuntimeState:
    published: int = 0
    reached: int = 0
    timeouts: int = 0


@dataclass
class BagPlaybackEvent:
    topic: str
    msg: HybridJointCommand


class BagInterpolatedPlayer(Node):
    """Read a ROS 2 bag and publish command topics with per-message interpolation.

    For each queued command message, the node keeps publishing interpolated positions
    toward that target (using live /joint_states feedback) and advances to the next
    queued message only after target reach (or timeout).
    """

    def __init__(self) -> None:
        super().__init__('bag_interpolated_player')

        self.declare_parameter('bag_path', '')
        self.declare_parameter('storage_id', '')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('wr1_topic', '/wr1_controller/commands')
        self.declare_parameter('hand_topic', '/hand_controller/commands')
        self.declare_parameter('wait_for_joint_state_sec', 2.0)

        # State-based playback params.
        self.declare_parameter('control_period_sec', 0.01)
        self.declare_parameter('joint_tolerance', 0.02)
        self.declare_parameter('max_step_per_cycle', 0.03)
        self.declare_parameter('reach_timeout_sec', 5.0)
        self.declare_parameter('wr1_kp_start_scale', 0.5)
        self.declare_parameter('wr1_kd_start_scale', 0.5)
        self.declare_parameter('wr1_gain_ramp_sec', 1.0)

        self.bag_path = str(self.get_parameter('bag_path').value)
        self.storage_id = str(self.get_parameter('storage_id').value)
        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.wr1_topic = str(self.get_parameter('wr1_topic').value)
        self.hand_topic = str(self.get_parameter('hand_topic').value)
        self.wait_for_joint_state_sec = float(self.get_parameter('wait_for_joint_state_sec').value)

        self.control_period_sec = float(self.get_parameter('control_period_sec').value)
        self.joint_tolerance = float(self.get_parameter('joint_tolerance').value)
        self.max_step_per_cycle = float(self.get_parameter('max_step_per_cycle').value)
        self.reach_timeout_sec = float(self.get_parameter('reach_timeout_sec').value)
        self.wr1_kp_start_scale = float(self.get_parameter('wr1_kp_start_scale').value)
        self.wr1_kd_start_scale = float(self.get_parameter('wr1_kd_start_scale').value)
        self.wr1_gain_ramp_sec = float(self.get_parameter('wr1_gain_ramp_sec').value)

        if not self.bag_path:
            raise RuntimeError('Parameter bag_path is required.')
        if self.control_period_sec <= 0.0:
            raise RuntimeError('Parameter control_period_sec must be > 0.')
        if self.joint_tolerance < 0.0:
            raise RuntimeError('Parameter joint_tolerance must be >= 0.')
        if self.max_step_per_cycle <= 0.0:
            raise RuntimeError('Parameter max_step_per_cycle must be > 0.')
        if self.wr1_kp_start_scale < 0.0 or self.wr1_kd_start_scale < 0.0:
            raise RuntimeError('wr1_kp_start_scale/wr1_kd_start_scale must be >= 0.')

        if not self.storage_id:
            self.storage_id = self._detect_storage_id(self.bag_path)

        self.latest_joint_positions: Dict[str, float] = {}
        self.seen_joint_state = False

        self.create_subscription(JointState, self.joint_state_topic, self._joint_state_cb, 100)
        self.wr1_pub = self.create_publisher(HybridJointCommand, self.wr1_topic, 50)
        self.hand_pub = self.create_publisher(HybridJointCommand, self.hand_topic, 50)

        self.get_logger().info(
            f'bag_path={self.bag_path}, storage_id={self.storage_id}, '
            f'period={self.control_period_sec:.3f}s, tol={self.joint_tolerance:.4f}, '
            f'max_step={self.max_step_per_cycle:.4f}, timeout={self.reach_timeout_sec:.2f}s, '
            f'wr1_kp_scale_start={self.wr1_kp_start_scale:.2f}, '
            f'wr1_kd_scale_start={self.wr1_kd_start_scale:.2f}, '
            f'wr1_gain_ramp_sec={self.wr1_gain_ramp_sec:.2f}'
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            self.latest_joint_positions[n] = p
        self.seen_joint_state = True

    def _detect_storage_id(self, bag_path: str) -> str:
        meta = os.path.join(bag_path, 'metadata.yaml')
        if os.path.isfile(meta):
            try:
                text = open(meta, 'r', encoding='utf-8').read()
                m = re.search(r'^\s*storage_identifier:\s*(\S+)\s*$', text, re.MULTILINE)
                if m:
                    return m.group(1)
            except Exception:
                pass
        return 'sqlite3'

    def _wait_for_joint_state(self) -> None:
        deadline = time.monotonic() + max(0.0, self.wait_for_joint_state_sec)
        while rclpy.ok() and time.monotonic() < deadline and not self.seen_joint_state:
            rclpy.spin_once(self, timeout_sec=0.05)

        if self.seen_joint_state:
            self.get_logger().info('Received live joint state snapshot.')
        else:
            self.get_logger().warn(
                f'No {self.joint_state_topic} received in {self.wait_for_joint_state_sec:.2f}s; '
                'missing joints fallback to target values.'
            )

    def _get_joint_names(self, msg: HybridJointCommand) -> List[str]:
        if hasattr(msg, 'joint_name'):
            return list(msg.joint_name)
        if hasattr(msg, 'name'):
            return list(msg.name)
        raise RuntimeError('HybridJointCommand has neither joint_name nor name field.')

    def _set_joint_names(self, msg: HybridJointCommand, names: List[str]) -> None:
        if hasattr(msg, 'joint_name'):
            msg.joint_name = list(names)
        elif hasattr(msg, 'name'):
            msg.name = list(names)
        else:
            raise RuntimeError('HybridJointCommand has neither joint_name nor name field.')

    def _copy_passthrough_fields(self, out: HybridJointCommand, src: HybridJointCommand) -> None:
        out.header = src.header
        self._set_joint_names(out, self._get_joint_names(src))
        out.velocity = list(src.velocity)
        out.kp = list(src.kp)
        out.kd = list(src.kd)

        if hasattr(src, 'feedforward') and hasattr(out, 'feedforward'):
            out.feedforward = list(src.feedforward)
        elif hasattr(src, 'effort') and hasattr(out, 'effort'):
            out.effort = list(src.effort)

    def _load_bag_events(self) -> Tuple[List[BagPlaybackEvent], Dict[str, TopicRuntimeState]]:
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=self.bag_path, storage_id=self.storage_id),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'),
        )

        wanted = {self.wr1_topic, self.hand_topic}
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        missing = [t for t in wanted if t not in types]
        if missing:
            raise RuntimeError(f'Missing expected topics in bag: {missing}')

        for t in wanted:
            if types[t] != 'xbot_common_interfaces/msg/HybridJointCommand':
                raise RuntimeError(f'Unexpected type on {t}: {types[t]}')

        states: Dict[str, TopicRuntimeState] = {
            self.wr1_topic: TopicRuntimeState(),
            self.hand_topic: TopicRuntimeState(),
        }
        events: List[BagPlaybackEvent] = []

        total_read = 0
        while rclpy.ok() and reader.has_next():
            topic, data, _ = reader.read_next()
            if topic not in wanted:
                continue
            cmd = deserialize_message(data, HybridJointCommand)
            events.append(BagPlaybackEvent(topic=topic, msg=cmd))
            total_read += 1

        self.get_logger().info(
            f'Loaded {total_read} command msgs from bag: '
            f'{self.wr1_topic}={sum(1 for e in events if e.topic == self.wr1_topic)}, '
            f'{self.hand_topic}={sum(1 for e in events if e.topic == self.hand_topic)}'
        )
        return events, states

    def _build_interpolated_step(self, msg: HybridJointCommand) -> Tuple[HybridJointCommand, bool]:
        names = self._get_joint_names(msg)
        out = HybridJointCommand()
        self._copy_passthrough_fields(out, msg)

        out_positions: List[float] = []
        reached = True

        for joint_name, target in zip(names, msg.position):
            current = self.latest_joint_positions.get(joint_name, target)
            err = target - current

            if abs(err) > self.joint_tolerance:
                reached = False

            if err > self.max_step_per_cycle:
                step = self.max_step_per_cycle
            elif err < -self.max_step_per_cycle:
                step = -self.max_step_per_cycle
            else:
                step = err

            out_positions.append(current + step)

        out.position = out_positions
        return out, reached

    def _apply_gain_ramp(self, msg: HybridJointCommand, elapsed_sec: float) -> None:
        if self.wr1_gain_ramp_sec <= 0.0:
            return
        alpha = max(0.0, min(1.0, elapsed_sec / self.wr1_gain_ramp_sec))
        kp_scale = self.wr1_kp_start_scale + (1.0 - self.wr1_kp_start_scale) * alpha
        kd_scale = self.wr1_kd_start_scale + (1.0 - self.wr1_kd_start_scale) * alpha
        msg.kp = [v * kp_scale for v in msg.kp]
        msg.kd = [v * kd_scale for v in msg.kd]

    def _log_gains(self, topic: str, msg: HybridJointCommand, tag: str) -> None:
        self.get_logger().info(
            f'{tag} {topic} kp={list(msg.kp)} kd={list(msg.kd)}'
        )

    def play(self) -> None:
        self._wait_for_joint_state()
        events, states = self._load_bag_events()

        self.get_logger().info(
            'Starting playback: preserving bag order, with state-based interpolation for wr1.'
        )

        next_event_idx = 0
        active_wr1_msg: Optional[HybridJointCommand] = None
        active_wr1_start_time_sec = 0.0

        while rclpy.ok():
            cycle_start = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.0)

            pending_any = active_wr1_msg is not None or next_event_idx < len(events)

            if active_wr1_msg is None and next_event_idx < len(events):
                event = events[next_event_idx]
                next_event_idx += 1

                if event.topic == self.hand_topic:
                    direct_msg = event.msg
                    self.hand_pub.publish(direct_msg)
                    states[self.hand_topic].published += 1
                    self._log_gains(event.topic, direct_msg, 'publish')
                else:
                    active_wr1_msg = event.msg
                    active_wr1_start_time_sec = time.monotonic()

            if active_wr1_msg is not None:
                out, reached = self._build_interpolated_step(active_wr1_msg)
                elapsed_for_msg = time.monotonic() - active_wr1_start_time_sec
                self._apply_gain_ramp(out, elapsed_for_msg)
                self.wr1_pub.publish(out)
                states[self.wr1_topic].published += 1
                self._log_gains(self.wr1_topic, out, 'publish')

                if reached:
                    # Send exact target once before advancing.
                    exact_msg = HybridJointCommand()
                    self._copy_passthrough_fields(exact_msg, active_wr1_msg)
                    exact_msg.position = list(active_wr1_msg.position)
                    self._apply_gain_ramp(exact_msg, elapsed_for_msg)
                    self.wr1_pub.publish(exact_msg)
                    states[self.wr1_topic].published += 1
                    self._log_gains(self.wr1_topic, exact_msg, 'publish_target')
                    states[self.wr1_topic].reached += 1
                    active_wr1_msg = None
                elif self.reach_timeout_sec > 0.0:
                    elapsed = time.monotonic() - active_wr1_start_time_sec
                    if elapsed >= self.reach_timeout_sec:
                        self.get_logger().warn(
                            f'Timeout waiting for target on {self.wr1_topic} after {elapsed:.2f}s; '
                            'advancing to next queued message.'
                        )
                        timeout_msg = HybridJointCommand()
                        self._copy_passthrough_fields(timeout_msg, active_wr1_msg)
                        timeout_msg.position = list(active_wr1_msg.position)
                        self._apply_gain_ramp(timeout_msg, elapsed)
                        self.wr1_pub.publish(timeout_msg)
                        states[self.wr1_topic].published += 1
                        self._log_gains(self.wr1_topic, timeout_msg, 'publish_timeout_target')
                        states[self.wr1_topic].timeouts += 1
                        active_wr1_msg = None

            if not pending_any:
                break

            elapsed = time.monotonic() - cycle_start
            sleep_sec = self.control_period_sec - elapsed
            if sleep_sec > 0.0:
                rclpy.spin_once(self, timeout_sec=sleep_sec)

        self.get_logger().info(
            f'Playback complete. '
            f'{self.wr1_topic}: published={states[self.wr1_topic].published}, '
            f'reached={states[self.wr1_topic].reached}, timeouts={states[self.wr1_topic].timeouts}; '
            f'{self.hand_topic}: published={states[self.hand_topic].published}, '
            f'reached={states[self.hand_topic].reached}, timeouts={states[self.hand_topic].timeouts}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = BagInterpolatedPlayer()
        node.play()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

