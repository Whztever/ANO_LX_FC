import datetime
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple

try:
    import pyrealsense2 as rs
    rs.config()
except:
    import pyrealsense2.pyrealsense2 as rs # for linux

from loguru import logger


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
    sinp = 2.0 * (w * y - z * x)
    p = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    r = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    y = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # convert radians to degrees
    r, p, y = math.degrees(r), math.degrees(p), math.degrees(y)
    return r, p, y


@dataclass
class T265_Pose_Frame(object):
    """
    T265 姿态数据帧
    """

    @dataclass
    class _XYZ:  # 三维坐标
        x: float
        y: float
        z: float

    @dataclass
    class _WXYZ:  # 四元数
        w: float
        x: float
        y: float
        z: float

    translation: _XYZ  # 位移 / m
    rotation: _WXYZ  # 四元数姿态
    velocity: _XYZ  # 速度 / m/s
    acceleration: _XYZ  # 加速度 / m/s^2
    angular_velocity: _XYZ  # 角速度 / rad/s
    angular_acceleration: _XYZ  # 角加速度 / rad/s^2
    tracker_confidence: int  # 跟踪置信度 0: Failed, 1: Low, 2: Medium, 3: High
    mapper_confidence: int  # 建图置信度 0: Failed, 1: Low, 2: Medium, 3: High


"""
note:
T265 姿态坐标系
            y
         z  ^
          \ |
           \|
   x<---[ (O O)]
pitch-dx, yaw-dy, roll-dz
所有轴向均为右手系
"""


class T265(object):
    """
    Realsense T265 包装类
    """

    def __init__(
        self, log_to_file: bool = False, log_to_console: bool = False, event_skip=0, log_level: str = "info", **args
    ) -> None:
        """
        初始化 T265
        """
        self.pose: T265_Pose_Frame = None  # type: ignore
        self.frame_num: int = 0  # frame number
        self.frame_timestamp: float = 0.0  # timestamp
        self.running = False
        self.update_event = threading.Event()
        self._event_skip = event_skip
        if log_to_file:
            rs.log_to_file(getattr(rs.log_severity, log_level), "rs_t265.log")
        if log_to_console:
            rs.log_to_console(getattr(rs.log_severity, log_level))
        self._connect(**args)
        self._connect_args = args
        self._cali_pos_offset = [0.0, 0.0, 0.0]
        self._cali_eular_offset = [0.0, 0.0, 0.0]
        self._callbacks: list[Callable] = []

    def _connect(self, **args) -> None:
        self._pipe = rs.pipeline()
        self._cfg = rs.config()
        self._cfg.enable_stream(rs.stream.pose)
        self._device = self._cfg.resolve(self._pipe).get_device()
        logger.info(f"Connected to {self._device}")
        logger.debug(f"Device sensors: {self._device.query_sensors()}")
        pose_sensor = self._device.first_pose_sensor()
        logger.debug(f"Pose sensor: {pose_sensor}")
        pose_sensor.set_option(rs.option.enable_auto_exposure, args.get("enable_auto_exposure", 1))
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
        if not pose:
            return
        self.frame_num = pose.frame_number
        self.frame_timestamp = pose.timestamp
        self.pose = pose.get_pose_data()
        if self._print_update:
            self.print_pose()
        self._update_count += 1
        if self._callbacks:
            for callback in self._callbacks:
                callback(self.pose, self.frame_num, self.frame_timestamp)
        if self._event_skip == 0 or self._update_count % self._event_skip == 0:
            self.update_event.set()

    def print_pose(self, refresh=True) -> None:
        BACK = "\033[F"
        r, p, y = self.eular_rotation
        text = (
            f"T265 Pose Frame #{self.frame_num} at {datetime.datetime.fromtimestamp(self.frame_timestamp / 1000)}\n"
            f"Translation    :{self.pose.translation.x:11.6f},{self.pose.translation.y:11.6f},{self.pose.translation.z:11.6f};\n"
            f"Velocity       :{self.pose.velocity.x:11.6f},{self.pose.velocity.y:11.6f},{self.pose.velocity.z:11.6f};\n"
            f"Acceleration   :{self.pose.acceleration.x:11.6f},{self.pose.acceleration.y:11.6f},{self.pose.acceleration.z:11.6f};\n"
            f"Angular vel    :{self.pose.angular_velocity.x:11.6f},{self.pose.angular_velocity.y:11.6f},{self.pose.angular_velocity.z:11.6f};\n"
            f"Angular accel  :{self.pose.angular_acceleration.x:11.6f},{self.pose.angular_acceleration.y:11.6f},{self.pose.angular_acceleration.z:11.6f};\n"
            f"Rotation       :{self.pose.rotation.w:11.6f},{self.pose.rotation.x:11.6f},{self.pose.rotation.y:11.6f},{self.pose.rotation.z:11.6f};\n"
            f"Roll/Pitch/Yaw :{r:11.6f},{p:11.6f},{y:11.6f};\n"
            f"Tracker conf: {self.pose.tracker_confidence}, Mapper conf: {self.pose.mapper_confidence}"
        )
        if refresh:
            print(f"{text}{BACK* 8}", end="")

    def start(self, async_update: bool = True, print_update: bool = False) -> None:
        """
        开始监听 T265

        async_update: 是否使用异步回调的方式监听, 若为 False, 则需要手动调用 update() 方法
        print_update: 是否在控制台打印更新
        lightweight_update: 是否使用轻量级更新(仅更新位置和四元数姿态数据)
        """
        self._async = async_update
        self._print_update = print_update
        if self._async:
            self._pipe.start(self._cfg, self._callback)
        else:
            self._pipe.start(self._cfg)
        self._update_count = 0
        self._start_time = time.perf_counter()
        self.running = True
        logger.info("T265 started")

    def update(self):
        """
        更新 T265 状态(阻塞直到有新的数据帧到来)
        """
        if not self.running:
            raise RuntimeError("T265 is not running")
        if self._async:
            raise RuntimeError("Async mode")
        frames = self._pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        if not pose:
            return
        self.frame_num = pose.frame_number
        self.frame_timestamp = pose.timestamp
        self.pose = pose.get_pose_data()
        if self._print_update:
            self.print_pose()
        self._update_count += 1
        if self._callbacks:
            for callback in self._callbacks:
                callback(self.pose, self.frame_num, self.frame_timestamp)
        if self._event_skip == 0 or self._update_count % self._event_skip == 0:
            self.update_event.set()

    def register_callback(self, callback: Callable[[T265_Pose_Frame, int, float], None]) -> None:
        """
        注册 T265 更新回调函数
        回调参数: T265_Pose_Frame, 帧编号, 帧时间戳(ms)
        """
        self._callbacks.append(callback)

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
        fps = self._update_count / (time.perf_counter() - self._start_time)
        self._update_count = 0
        self._start_time = time.perf_counter()
        return fps

    def hardware_reset(self) -> None:
        """
        强制重置 T265 并重新连接
        """
        self._device.hardware_reset()
        logger.warning("T265 hardware reset, waiting for reconnection...")
        while True:
            try:
                self._connect(**self._connect_args)
                break
            except RuntimeError:
                time.sleep(1)
        if self.running:
            if self._async:
                self._pipe.start(self._cfg, self._callback)
            else:
                self._pipe.start(self._cfg)
            self._update_count = 0
            self._start_time = time.perf_counter()

    def calibrate_pos(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        """
        软件计算位移偏移值
        x, y, z: 实际位置
        """
        self._cali_pos_offset = [x - self.pose.translation.x, y - self.pose.translation.y, z - self.pose.translation.z]
        logger.info(f"Calibration offset: {self._cali_pos_offset}")

    @property
    def calibrated_pos(self) -> Tuple[float, float, float]:
        """
        获取校准后的位置
        """
        return (
            self.pose.translation.x + self._cali_pos_offset[0],
            self.pose.translation.y + self._cali_pos_offset[1],
            self.pose.translation.z + self._cali_pos_offset[2],
        )

    def calibrate_eular(self, r: float = 0, p: float = 0, y: float = 0) -> None:
        """
        软件计算欧拉角偏移值
        r, p, y: 实际欧拉角
        """
        tr, tp, ty = self.eular_rotation
        self._cali_eular_offset = [r - tr, p - tp, y - ty]
        logger.info(f"Calibration offset: {self._cali_eular_offset}")

    @property
    def calibrated_eular(self) -> Tuple[float, float, float]:
        """
        获取校准后的欧拉角
        """

        def recomp(e, size) -> float:
            if e > size:
                return e - 2 * size
            elif e < -size:
                return e + 2 * size
            else:
                return e

        tr, tp, ty = self.eular_rotation
        return (
            recomp(tr + self._cali_eular_offset[0], 180),
            recomp(tp + self._cali_eular_offset[1], 90),
            recomp(ty + self._cali_eular_offset[2], 180),
        )

    @property
    def eular_rotation(self) -> Tuple[float, float, float]:
        """
        获取欧拉角姿态
        返回值: roll, pitch, yaw
        """
        # in convert matrices: roll (x), pitch (y), yaw (z)
        # so we swap axis: x, y, z = r_z, r_x, r_y
        return quaternions_to_euler(
            self.pose.rotation.w, self.pose.rotation.z, self.pose.rotation.x, self.pose.rotation.y
        )


if __name__ == "__main__":
    t265 = T265()
    t265.hardware_reset()
    t265.start(print_update=True)
    try:
        while True:
            time.sleep(20)
            # t265.hardware_reset()
    finally:
        t265.stop()
