import math
import time
from typing import Any, Tuple

import pyrealsense2 as rs
from loguru import logger

BACK = "\033[F"


def quaternions_to_euler(w, x, y, z):
    # mathod 1
    # r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # p = math.asin(2 * (w * y - z * x))
    # y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # mathod 2
    # r = math.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    # p = -math.asin(2.0 * (x * z - w * y))
    # y = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)

    # mathod 3
    # Resolve the gimbal lock problem
    r = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    p = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    y = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # convert axis and radians to degrees
    r, p, y = -y / math.pi * 180, r / math.pi * 180, p / math.pi * 180
    return r, p, y


class T265_State(object):
    p_x: float = 0.0  # translation x / m
    p_y: float = 0.0  # translation y / m
    p_z: float = 0.0  # translation z / m
    v_x: float = 0.0  # velocity x / m/s
    v_y: float = 0.0  # velocity y / m/s
    v_z: float = 0.0  # velocity z / m/s
    a_x: float = 0.0  # acceleration x / m/s^2
    a_y: float = 0.0  # acceleration y / m/s^2
    a_z: float = 0.0  # acceleration z / m/s^2
    av_x: float = 0.0  # angular velocity x / rad/s
    av_y: float = 0.0  # angular velocity y / rad/s
    av_z: float = 0.0  # angular velocity z / rad/s
    aa_x: float = 0.0  # angular acceleration x / rad/s^2
    aa_y: float = 0.0  # angular acceleration y / rad/s^2
    aa_z: float = 0.0  # angular acceleration z / rad/s^2
    r_w: float = 0.0  # rotation w / quaternion
    r_x: float = 0.0  # rotation x / quaternion
    r_y: float = 0.0  # rotation y / quaternion
    r_z: float = 0.0  # rotation z / quaternion
    track_conf: int = 0  # tracker confidence
    mapper_conf: int = 0  # mapper confidence
    frame_num: int = 0  # frame number
    timestamp: float = 0.0  # timestamp

    def update(
        self, data: Any, as_pose_frame: bool = False, as_frames: bool = False, lightweight: bool = False
    ) -> None:
        if as_frames:
            data = data.get_pose_frame()
            as_pose_frame = True
        if as_pose_frame:
            if not data:
                return
            if not lightweight:
                self.timestamp = data.timestamp
                self.frame_num = data.frame_number
            data = data.get_pose_data()
        self.p_x, self.p_y, self.p_z = data.translation.x, data.translation.y, data.translation.z
        self.r_w, self.r_x, self.r_y, self.r_z = data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z
        if not lightweight:
            self.v_x, self.v_y, self.v_z = data.velocity.x, data.velocity.y, data.velocity.z
            self.a_x, self.a_y, self.a_z = data.acceleration.x, data.acceleration.y, data.acceleration.z
            self.av_x, self.av_y, self.av_z = data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z
            self.aa_x, self.aa_y, self.aa_z = (
                data.angular_acceleration.x,
                data.angular_acceleration.y,
                data.angular_acceleration.z,
            )
            self.track_conf, self.mapper_conf = data.tracker_confidence, data.mapper_confidence

    @property
    def euler(self) -> tuple[float, float, float]:
        return quaternions_to_euler(self.r_w, self.r_x, self.r_y, self.r_z)

    def __str__(self):
        r, p, y = self.euler
        return (
            f"T265 Pose Frame #{self.frame_num} at {self.timestamp}:\n"
            f"Position       :{self.p_x:10.6f}, {self.p_y:10.6f}, {self.p_z:10.6f};\n"
            f"Velocity       :{self.v_x:10.6f}, {self.v_y:10.6f}, {self.v_z:10.6f};\n"
            f"Acceleration   :{self.a_x:10.6f}, {self.a_y:10.6f}, {self.a_z:10.6f};\n"
            f"Angular vel    :{self.av_x:10.6f}, {self.av_y:10.6f}, {self.av_z:10.6f};\n"
            f"Angular accel  :{self.aa_x:10.6f}, {self.aa_y:10.6f}, {self.aa_z:10.6f};\n"
            f"Rotation       :{self.r_w:10.6f}, {self.r_x:10.6f}, {self.r_y:10.6f}, {self.r_z:10.6f};\n"
            f"Roll/Pitch/Yaw :{r:10.5f}, {p:10.5f}, {y:10.5f};\n"
            f"Tracker confidence: {self.track_conf}, Mapper confidence: {self.mapper_conf}"
        )


class T265(object):
    """
    Realsense T265 包装类
    """

    def __init__(
        self, log_to_file: bool = False, log_to_console: bool = False, log_level: str = "info", **args
    ) -> None:
        """
        初始化 T265
        """
        self.state = T265_State()
        self.running = False
        if log_to_file:
            rs.log_to_file(getattr(rs.log_severity, log_level), "rs_t265.log")
        if log_to_console:
            rs.log_to_console(getattr(rs.log_severity, log_level))
        self._connect(**args)
        self._connect_args = args
        self._cali_offset = [0.0, 0.0, 0.0]

    def _connect(self, **args) -> None:
        self._pipe = rs.pipeline()
        self._cfg = rs.config()
        self._cfg.enable_stream(rs.stream.pose)
        # self._cfg.enable_stream(rs.stream.fisheye, 1)  # left
        # self._cfg.enable_stream(rs.stream.fisheye, 2)  # right
        # self._cfg.enable_stream(rs.stream.accel)
        # self._cfg.enable_stream(rs.stream.gyro)
        self._device = self._cfg.resolve(self._pipe).get_device()
        logger.info(f"Connected to {self._device}")
        logger.debug(f"Device sensors: {self._device.query_sensors()}")
        pose_sensor = self._device.first_pose_sensor()
        logger.debug(f"Pose sensor: {pose_sensor}")
        pose_sensor.set_option(rs.option.enable_mapping, args.get("enable_mapping", 1))
        pose_sensor.set_option(rs.option.enable_map_preservation, args.get("enable_map_preservation", 1))
        pose_sensor.set_option(rs.option.enable_relocalization, args.get("enable_relocalization", 1))
        pose_sensor.set_option(rs.option.enable_pose_jumping, args.get("enable_pose_jumping", 1))
        pose_sensor.set_option(rs.option.enable_dynamic_calibration, args.get("enable_dynamic_calibration", 1))
        logger.debug(f"Pose sensor options:")
        for opt in pose_sensor.get_supported_options():
            logger.debug(f"  {opt}: {pose_sensor.get_option(opt)}")

    def _callback(self, frame) -> None:
        pose = frame.as_pose_frame()
        self.state.update(pose, as_pose_frame=True, lightweight=self._lightweight_update)
        if self._print_update:
            print(f"{self.state}{BACK* 8}", end="")

    def start(self, async_: bool = True, print_update: bool = False, lightweight_update: bool = False) -> None:
        """
        开始监听 T265
        async_: 是否使用回调的方式监听, 若为 False, 则需要手动调用 update() 方法
        print_update: 是否在控制台打印更新
        lightweight_update: 是否使用轻量级更新(仅更新位置和四元数姿态数据)
        """
        self._async = async_
        self._print_update = print_update
        self._lightweight_update = lightweight_update
        if self._async:
            self._pipe.start(self._cfg, self._callback)
        else:
            self._pipe.start(self._cfg)
        self.start_time = time.perf_counter()
        self.running = True
        logger.info("T265 started")

    def update(self):
        """
        更新 T265 状态
        """
        if not self.running:
            raise RuntimeError("T265 is not running")
        if self._async:
            raise RuntimeError("It's not necessary to call update() when async is True")
        frames = self._pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        self.state.update(pose, as_pose_frame=True, lightweight=self._lightweight_update)
        if self._print_update:
            print(f"{self.state}{BACK* 8}", end="")

    def stop(self) -> None:
        """
        停止监听 T265
        """
        self._pipe.stop()
        self.running = False
        logger.info("T265 stopped")

    @property
    def fps(self) -> float:
        """
        获取 T265 的平均更新速率
        """
        return self.state.frame_num / (time.perf_counter() - self.start_time)

    def hardware_reset(self) -> None:
        """
        强制重置 T265
        """
        self._device.hardware_reset()
        logger.warning("T265 hardware reset, waiting for reconnection...")
        while True:
            try:
                self._connect(**self._connect_args)
                break
            except RuntimeError:
                time.sleep(1)

    def calibrate(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        """
        软件计算偏移值
        x, y, z: 实际位置
        """
        self._cali_offset = [x - self.state.p_x, y - self.state.p_y, z - self.state.p_z]
        logger.info(f"Calibration offset: {self._cali_offset}")

    @property
    def calibrated_pos(self) -> Tuple[float, float, float]:
        """
        获取校准后的位置
        """
        return (
            self.state.p_x + self._cali_offset[0],
            self.state.p_y + self._cali_offset[1],
            self.state.p_z + self._cali_offset[2],
        )


if __name__ == "__main__":
    t265 = T265()
    # t265.hardware_reset()
    t265.start(print_update=True)
    try:
        # while True:
        #     time.sleep(1)
        time.sleep(5)
        t265.calibrate(0, 0, 0)
        time.sleep(10)
    finally:
        t265.stop()
