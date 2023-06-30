import threading
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from FlightController import FC_Client, FC_Controller
from FlightController.Components import LD_Radar
from FlightController.Components.RealSense import T265
from loguru import logger
from simple_pid import PID


class Navigation(object):
    """
    闭环导航, 使用realsense T265作为位置闭环, 使用雷达作为定位校准, 使用飞控返回的激光高度作为高度闭环
    """

    def __init__(self, fc: Union[FC_Client, FC_Controller], radar: LD_Radar, rs: T265):
        self.fc = fc
        self.radar = radar
        self.rs = rs
        ############### PID #################
        self.pid_tunings = {
            "default": (0.35, 0.0, 0.08),  # 导航
            "landing": (0.4, 0.05, 0.16),  # 降落
        }  # PID参数 (仅导航XY使用)
        self.height_pid = PID(0.8, 0.0, 0.1, setpoint=0, output_limits=(-30, 30), auto_mode=False)
        self.navi_x_pid = PID(
            *self.pid_tunings["default"],
            setpoint=0,
            output_limits=(-30, 30),
            auto_mode=False,
        )
        self.navi_y_pid = PID(
            *self.pid_tunings["default"],
            setpoint=0,
            output_limits=(-30, 30),
            auto_mode=False,
        )
        self.yaw_pid = PID(
            0.7,
            0.0,
            0.05,
            setpoint=0,
            output_limits=(-30, 30),
            auto_mode=False,
        )
        #####################################
        self.current_x = 0  # 当前位置X(相对于基地点) / cm
        self.current_y = 0  # 当前位置Y(相对于基地点) / cm
        self.current_yaw = 0  # 当前偏航角(顺时针为正) / deg
        self.current_height = 0  # 当前高度(激光高度) / cm
        self.basepoint = np.array([0.0, 0.0])  # 基地点(雷达坐标系)
        #####################################
        self.keep_height_flag = False  # 定高状态
        self.navigation_flag = False  # 导航状态
        self.running = False
        self.debug = False  # 调试模式
        self._thread_list: List[threading.Thread] = []

    def reset_basepoint(self) -> np.ndarray:
        """
        重置基地点到当前雷达位置
        """
        time.sleep(1)  # 等待雷达数据稳定
        if not self.radar.rt_pose_update_event.wait(3):
            logger.error("[NAVI] reset_basepoint(): Radar pose update timeout")
            raise RuntimeError("Radar pose update timeout")
        x, y, _ = self.radar.rt_pose
        self.basepoint = np.array([x, y])
        logger.info(f"[NAVI] Basepoint reset to {self.basepoint}")
        return self.basepoint

    def set_basepoint(self, point):
        """
        设置基地点(雷达坐标系)
        """
        self.basepoint = np.asarray(point)
        logger.info(f"[NAVI] Basepoint set to {self.basepoint}")

    def set_navigation_state(self, state: bool):
        """
        设置导航状态
        """
        self.navigation_flag = state
        if state and self.fc.state.mode.value != self.fc.HOLD_POS_MODE:
            self.fc.set_flight_mode(self.fc.HOLD_POS_MODE)
            logger.debug("[NAVI] Auto set fc mode to HOLD_POS_MODE")

    def set_keep_height_state(self, state: bool):
        """
        设置定高状态
        """
        self.keep_height_flag = state

    def stop(self, join=False):
        """
        停止导航
        """
        self.running = False
        self.fc.stop_realtime_control()
        self.radar.stop_resolve_pose()
        if join:
            for thread in self._thread_list:
                thread.join()
        logger.info("[NAVI] Navigation stopped")

    def start(self):
        """
        启动导航
        """
        ######## 解算参数 ########
        SIZE = 1000
        SCALE_RATIO = 0.5
        LOW_PASS_RATIO = 0.6
        RADAR_SKIP = 20  # 400/RADAR_SKIP
        RS_SKIP = 5  # 200/RS_SKIP
        ########################
        if self.running:
            logger.warning("[NAVI] Navigation already running, restarting...")
            self.stop(join=True)
        assert self.rs.running, "RealSense not running"
        assert self.radar.running, "Radar not running"
        self.running = True
        self.radar.subtask_skip = RADAR_SKIP
        self.rs.event_skip = RS_SKIP
        self.radar.start_resolve_pose(
            size=SIZE,
            scale_ratio=SCALE_RATIO,
            low_pass_ratio=LOW_PASS_RATIO,
        )
        logger.info("[NAVI] Resolve pose started")
        self.fc.update_realtime_control(vel_x=0, vel_y=0, vel_z=0, yaw=0)
        self.fc.start_realtime_control(40)
        logger.info("[NAVI] Realtime control started")
        self._thread_list.append(threading.Thread(target=self.keep_height_task, daemon=True))
        self._thread_list[-1].start()
        self._thread_list.append(threading.Thread(target=self.navigation_task, daemon=True))
        self._thread_list[-1].start()
        logger.info("[NAVI] Navigation started")

    def switch_pid(self, pid: Union[str, tuple]):
        """
        切换PID参数

        pid: str-在self.pid_tunings中的键值, tuple-自定义PID参数
        """
        if isinstance(pid, str):
            tuning = self.pid_tunings.get(pid, self.pid_tunings["default"])
        else:
            tuning = pid  # type: ignore
        self.navi_x_pid.tunings = tuning
        self.navi_y_pid.tunings = tuning
        logger.debug(f"[NAVI] PID Tunings set to {pid}: {tuning}")

    def _keep_height_task(self):
        paused = False
        while self.running:
            if not self.fc.state.update_event.wait(1):
                logger.warning("[NAVI] FC state update timeout")
                self.fc.update_realtime_control(vel_z=0)
                continue
            self.fc.state.update_event.clear()
            self.current_height = self.fc.state.alt_add.value
            if self.debug:
                logger.debug(f"[NAVI] Current height: {self.current_height}")
            if not (
                self.keep_height_flag
                and self.fc.state.mode.value == self.fc.HOLD_POS_MODE
                and self.fc.state.unlock.value
            ):
                if not paused:
                    paused = True
                    self.height_pid.set_auto_mode(False)
                    self.fc.update_realtime_control(vel_z=0)
                    logger.info("[NAVI] Keep height paused")
                continue
            if paused:
                paused = False
                self.height_pid.set_auto_mode(True, last_output=0)
                logger.info("[NAVI] Keep Height resumed")
            out_hei = round(self.height_pid(self.current_height))
            self.fc.update_realtime_control(vel_z=out_hei)
            logger.debug(f"[NAVI] Height PID output: {out_hei}")

    def _navigation_task(self):
        paused = False
        while self.running:
            time.sleep(0.001)
            if not self.rs.update_event.wait(1):  # 等待地图更新
                logger.warning("[NAVI] RealSense pose timeout")
                self.fc.update_realtime_control(vel_x=0, vel_y=0, yaw=0)
                continue
            self.rs.update_event.clear()
            if not self.rs.secondary_frame_established:
                self.current_x = self.basepoint[0] - self.rs.pose.translation.z * 100
                self.current_y = self.basepoint[1] - self.rs.pose.translation.x * 100
                self.current_yaw = -self.rs.eular_rotation[2]
            else:
                position, eular = self.rs.get_pose_in_secondary_frame(as_eular=True)
                self.current_x = self.basepoint[0] - position[2] * 100
                self.current_y = self.basepoint[1] - position[0] * 100
                self.current_yaw = -eular[2]
            available = self.rs.pose.tracker_confidence >= 2
            if self.debug:  # debug
                logger.debug(
                    (
                        f"[NAVI] T265 pose: {self.current_x}, {self.current_y}, {self.current_yaw}; "
                        f"Radar pose: {self.radar.rt_pose[0]}, {self.radar.rt_pose[1]}, {self.radar.rt_pose[2]}"
                    )
                )
            if not (
                self.navigation_flag
                and self.fc.state.mode.value == self.fc.HOLD_POS_MODE
                and self.fc.state.unlock.value
            ):  # 导航需在解锁/定点模式下运行
                if not paused:
                    paused = True
                    self.navi_x_pid.set_auto_mode(False)
                    self.navi_y_pid.set_auto_mode(False)
                    self.yaw_pid.set_auto_mode(False)
                    self.fc.update_realtime_control(vel_x=0, vel_y=0, yaw=0)
                    logger.info("[NAVI] Navigation paused")
                continue
            if paused:
                paused = False
                self.navi_x_pid.set_auto_mode(True, last_output=0)
                self.navi_y_pid.set_auto_mode(True, last_output=0)
                self.yaw_pid.set_auto_mode(True, last_output=0)
                logger.info("[NAVI] Navigation resumed")
            if not available:
                logger.warning("[NAVI] Pose from T265 not available")
                time.sleep(0.1)
                continue
            # self.fc.send_general_position(x=self.current_x, y=self.current_y)
            out_x = round(self.navi_x_pid(self.current_x))
            if out_x is not None:
                self.fc.update_realtime_control(vel_x=out_x)
            out_y = round(self.navi_y_pid(self.current_y))
            if out_y is not None:
                self.fc.update_realtime_control(vel_y=out_y)
            out_yaw = round(self.yaw_pid(self.current_yaw))
            if out_yaw is not None:
                self.fc.update_realtime_control(yaw=out_yaw)
            if self.debug:  # debug
                logger.debug(f"[NAVI] Pose PID output: {out_x}, {out_y}, {out_yaw}")

    def calibrate_realsense(self):
        """
        根据雷达数据校准T265的副坐标系
        """
        time.sleep(1)  # 等待雷达数据稳定
        if not self.radar.rt_pose_update_event.wait(3):
            logger.error("[NAVI] calibrate_realsense(): Radar pose update timeout")
            raise RuntimeError("Radar pose update timeout")
        x, y, yaw = self.radar.rt_pose
        z = self.fc.state.alt_add.value
        dx = x - self.basepoint[0]  # -> t265 -z * 100
        dx = -dx / 100.0
        dy = y - self.basepoint[1]  # -> t265 -x * 100
        dy = -dy / 100.0
        dz = z / 100.0
        dyaw = -yaw
        logger.debug(f"[NAVI] Calibrate T265: dz={dx}, dx={dy}, dy={dz}, dyaw={dyaw}")
        self.rs.establish_secondary_origin(force_level=True, z_offset=dx, x_offset=dy, y_offset=dz, yaw_offset=dyaw)

    def navigation_to_waypoint(self, waypoint):
        """
        导航到指定的目标点

        waypoint: (x, y) 相对于基地点的坐标 / cm
        """
        self.navi_x_pid.setpoint = waypoint[0]
        self.navi_y_pid.setpoint = waypoint[1]
        logger.debug(f"[NAVI] Navigation to waypoint: {waypoint}")

    @property
    def navigation_target(self) -> np.ndarray:
        """
        当前导航目标点
        """
        waypoint = np.array([self.navi_x_pid.setpoint, self.navi_y_pid.setpoint])
        return waypoint

    @navigation_target.setter
    def navigation_target(self, waypoint: np.ndarray):
        return self.navigation_to_waypoint(waypoint)

    def navigation_stop_here(self) -> np.ndarray:
        """
        原地停止(设置目标点为当前位置)

        return: 原定目标点
        """
        waypoint = self.navigation_target
        self.navi_x_pid.setpoint = self.current_x
        self.navi_y_pid.setpoint = self.current_y
        logger.debug(f"[NAVI] Navigation stopped at {self.current_x}, {self.current_y}")
        return waypoint

    def set_height(self, height: float):
        """
        设置飞行高度

        height: 激光高度 / cm
        """
        self.height_pid.setpoint = height
        logger.debug(f"[NAVI] Keep height set to {height}")

    def set_yaw(self, yaw: float):
        """
        设置飞行航向

        yaw: 相对于初始状态的航向角 / deg
        """
        self.yaw_pid.setpoint = yaw
        logger.debug(f"[NAVI] Keep yaw set to {yaw}")

    def navigation_to_waypoint_relative(self, waypoint_rel):
        """
        导航到指定的目标点

        waypoint_rel: (x, y) 相对偏移于当前位置的坐标 / cm
        """
        self.navi_x_pid.setpoint += waypoint_rel[0]
        self.navi_y_pid.setpoint += waypoint_rel[1]
        logger.debug(
            f"[NAVI] Navigation to waypoint: {self.navi_x_pid.setpoint}, {self.navi_y_pid.setpoint} (relative from {waypoint_rel})"
        )

    def set_navigation_speed(self, speed):
        """
        设置导航速度

        speed: 速度 / cm/s
        """
        speed = abs(speed)
        self.navi_x_pid.output_limits = (-speed, speed)
        self.navi_y_pid.output_limits = (-speed, speed)
        logger.info(f"[NAVI] Navigation speed set to {speed}")

    def set_vertical_speed(self, speed):
        """
        设置垂直速度

        speed: 速度 / cm/s
        """
        speed = abs(speed)
        self.height_pid.output_limits = (-speed, speed)
        logger.info(f"[NAVI] Vertical speed set to {speed}")

    def set_yaw_speed(self, speed):
        """
        设置偏航速度

        speed: 速度 / deg/s
        """
        speed = abs(speed)
        self.yaw_pid.output_limits = (-speed, speed)
        logger.info(f"[NAVI] Yaw speed set to {speed}")

    def _reached_waypoint(self, pos_thres=15):
        return (
            abs(self.current_x - self.navi_x_pid.setpoint) < pos_thres
            and abs(self.current_y - self.navi_y_pid.setpoint) < pos_thres
        )

    def pointing_takeoff(self, point, target_height=140):
        """
        定点起飞

        point: (x, y) 相对于基地点的坐标 / cm
        target_height: 起飞高度 / cm
        """
        logger.info(f"[NAVI] Takeoff at {point}")
        self.navigation_flag = False
        self.keep_height_flag = False
        self.fc.set_flight_mode(self.fc.PROGRAM_MODE)
        self.fc.unlock()
        inital_yaw = self.fc.state.yaw.value
        time.sleep(2)  # 等待电机启动
        self.fc.take_off(80)
        self.fc.wait_for_takeoff_done()
        self.fc.set_yaw(inital_yaw, 25)
        self.fc.wait_for_hovering(2)
        ######## 闭环定高
        self.fc.set_flight_mode(self.fc.HOLD_POS_MODE)
        self.set_height(target_height)
        self.keep_height_flag = True
        self.wait_for_height()
        self.navigation_to_waypoint(point)  # 初始化路径点
        self.switch_pid("default")
        time.sleep(0.1)
        self.navigation_flag = True
        self.set_navigation_speed(self.navigation_speed)

    def pointing_landing(self, point):
        """
        定点降落

        point: (x, y) 相对于基地点的坐标 / cm
        """
        logger.info(f"[NAVI] Landing at {point}")
        self.navigation_flag = True
        self.keep_height_flag = True
        self.navigation_to_waypoint(point)
        self.wait_for_waypoint()
        self.set_navigation_speed(self.precision_speed)
        self.switch_pid("landing")
        time.sleep(0.5)
        self.set_height(60)
        self.wait_for_height()
        self.set_height(30)
        time.sleep(1.5)
        self.set_height(20)
        time.sleep(2)
        self.wait_for_waypoint()
        self.set_height(0)
        # self.fc.land()
        self.fc.wait_for_lock(5)
        self.fc.lock()
        self.navigation_flag = False
        self.keep_height_flag = False

    def wait_for_waypoint(self, time_thres=0.5, pos_thres=15, timeout=30):
        """
        等待到达目标点

        time_thres: 到达目标点后积累的时间/s
        pos_thres: 到达目标点的距离阈值/cm
        timeout: 超时时间/s
        """
        time_count = 0
        time_start = time.perf_counter()
        while True:
            time.sleep(0.05)
            if self._reached_waypoint(pos_thres):
                time_count += 0.05
            if time_count >= time_thres:
                logger.info("[NAVI] Reached waypoint")
                return
            if time.perf_counter() - time_start > timeout:
                logger.warning("[NAVI] Waypoint overtime")
                return

    def wait_for_height(self, time_thres=0.5, height_thres=8, timeout=30):
        """
        等待到达目标高度(定高设定值)

        time_thres: 到达目标高度后积累的时间/s
        pos_thres: 到达目标高度的阈值/cm
        timeout: 超时时间/s
        """
        time_start = time.perf_counter()
        time_count = 0
        while True:
            time.sleep(0.05)
            if abs(self.current_height - self.height_pid.setpoint) < height_thres:
                time_count += 0.05
            if time_count >= time_thres:
                logger.info("[NAVI] Reached height")
                return
            if time.perf_counter() - time_start > timeout:
                logger.warning("[NAVI] Height overtime")
                return

    def wait_for_yaw(self, time_thres=0.5, yaw_thres=5, timeout=30):
        """
        等待到达目标偏航角

        time_thres: 到达目标偏航角后积累的时间/s
        pos_thres: 到达目标偏航角的阈值/deg
        timeout: 超时时间/s
        """
        time_start = time.perf_counter()
        time_count = 0
        while True:
            time.sleep(0.05)
            if abs(self.current_yaw - self.yaw_pid.setpoint) < yaw_thres:
                time_count += 0.05
            if time_count >= time_thres:
                logger.info("[NAVI] Reached yaw")
                return
            if time.perf_counter() - time_start > timeout:
                logger.warning("[NAVI] Yaw overtime")
                return

    def wait_for_waypoint_with_avoidance(self, time_thres=1, pos_thres=15, timeout=60):
        """
        等待到达目标点，同时进行避障

        time_thres: 到达目标点后积累的时间/s
        pos_thres: 到达目标点的距离阈值/cm
        timeout: 超时时间/s
        """
        time_count = 0
        time_start = time.perf_counter()
        while True:
            time.sleep(0.05)
            if self._reached_waypoint(pos_thres):
                time_count += 0.05
            if time_count >= time_thres:
                logger.info("[NAVI] Reached waypoint")
                return
            if time.perf_counter() - time_start > timeout:
                logger.warning("[NAVI] Waypoint overtime")
                return
            self._avoidance_handler()

    def _avoidance_handler(self):
        points = self.radar.map.find_nearest(self._avd_fp_from, self._avd_fp_to, 1, self._avd_fp_dist)
        if len(points) > 0:
            logger.warning(f"[NAVI] Found obstacle: {points[0]}")
            waypoint = self.navigation_stop_here()  # 原地停下
            self.set_height(self._avd_height)
            self.fc.set_rgb_led(255, 0, 0)
            time.sleep(1)
            self.fc.set_rgb_led(0, 0, 0)
            self.wait_for_height()
            self.keep_height_flag = False
            self.navigation_flag = False
            self.fc.set_flight_mode(self.fc.PROGRAM_MODE)
            self.fc.horizontal_move(self._avd_move + points[0].distance / 10, 25, self._avd_deg)
            self.fc.set_rgb_led(255, 255, 0)
            self.fc.wait_for_last_command_done()
            self.fc.set_rgb_led(0, 0, 0)
            self.navigation_stop_here()
            self.fc.set_flight_mode(self.fc.HOLD_POS_MODE)
            self.navigation_flag = True
            self.keep_height_flag = True
            self.height_pid.setpoint = self.default_takeoff_height
            self.wait_for_height()
            self.navigation_to_waypoint(waypoint)

    def set_avoidance_args(
        self,
        deg: int = 0,
        deg_range: int = 30,
        dist: int = 600,
        avd_height: int = 200,
        avd_move: int = 150,
    ):
        """
        设置避障参数

        fp_deg: 目标避障角度(deg) (0~360)
        fp_deg_range: 目标避障角度范围(deg)
        fp_dist: 目标避障距离(mm)
        avd_height: 避障目标高度(cm)
        avd_move: 避障移动距离(cm)
        """
        self._avd_fp_from = deg - deg_range
        self._avd_fp_to = deg + deg_range
        self._avd_fp_dist = dist
        self._avd_height = avd_height
        self._avd_deg = deg
        self._avd_move = avd_move
