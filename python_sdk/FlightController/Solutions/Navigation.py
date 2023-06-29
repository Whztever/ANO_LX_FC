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
    def __init__(self, fc: Union[FC_Client, FC_Controller], radar: LD_Radar, rs: T265):
        self.fc = fc
        self.radar = radar
        self.rs = rs
        ############### PID #################
        self.pid_tunings = {
            "default": (0.35, 0.0, 0.08),  # 导航
        }  # PID参数 (仅导航XY使用)
        self.height_pid = PID(0.8, 0.0, 0.1, setpoint=0, output_limits=(-30, 30), auto_mode=False)
        self.navi_x_pid = PID(
            0.4,
            0,
            0.08,
            setpoint=0,
            output_limits=(-0.01, 0.01),
            auto_mode=False,
        )
        self.navi_y_pid = PID(
            0.4,
            0,
            0.08,
            setpoint=0,
            output_limits=(-0.01, 0.01),
            auto_mode=False,
        )
        self.navi_yaw_pid = PID(
            0.7,
            0.0,
            0.0,
            setpoint=0,
            output_limits=(-30, 30),
            auto_mode=False,
        )
        self.current_x = 0
        self.current_y = 0
        self.current_yaw = 0
        self.current_height = 0
        self.basepoint = np.array([0.0, 0.0])
        #####################################
        self.keep_height_flag = False
        self.navigation_flag = False
        self.running = False
        self._thread_list: List[threading.Thread] = []

    def reset_basepoint(self):
        if not self.radar.rt_pose_update_event.wait(3):
            logger.error("[NAVI] reset_basepoint(): Radar pose update timeout")
            raise RuntimeError("Radar pose update timeout")
        x, y, _ = self.radar.rt_pose
        self.basepoint = np.array([0.0, 0.0])
        logger.info(f"[NAVI] Basepoint reset to {self.basepoint}")

    def set_navigation_state(self, state: bool):
        self.navigation_flag = state
        logger.info(f"[NAVI] Navigation state set to {state}")

    def set_keep_height_state(self, state: bool):
        self.keep_height_flag = state
        logger.info(f"[NAVI] Keep height state set to {state}")

    def stop(self, join=False):
        self.running = False
        self.fc.stop_realtime_control()
        if join:
            for thread in self._thread_list:
                thread.join()
        logger.info("[NAVI] Threads stopped")
        self.radar.stop_resolve_pose()

    def start(self):
        ######## 解算参数 ########
        SIZE = 1000
        SCALE_RATIO = 0.5
        LOW_PASS_RATIO = 0.6
        RADAR_SKIP = 10
        RS_SKIP = 10
        ########################
        self.running = True
        self.radar.subtask_skip = RADAR_SKIP
        self.rs.event_skip = RS_SKIP
        self._thread_list.append(threading.Thread(target=self.keep_height_task, daemon=True))
        self._thread_list[-1].start()
        self._thread_list.append(threading.Thread(target=self.navigation_task, daemon=True))
        self._thread_list[-1].start()
        logger.info("[NAVI] Threads started")
        self.radar.start_resolve_pose(
            size=SIZE,
            scale_ratio=SCALE_RATIO,
            low_pass_ratio=LOW_PASS_RATIO,
        )
        logger.info("[NAVI] Resolve pose started")
        self.fc.start_realtime_control(20)
        logger.info("[NAVI] Realtime control started")

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
            time.sleep(1 / 12)  # 飞控参数以20Hz回传
            if self.keep_height_flag and self.fc.state.mode.value == self.fc.HOLD_POS_MODE:
                if paused:
                    paused = False
                    self.height_pid.set_auto_mode(True, last_output=0)
                    logger.info("[NAVI] Keep Height resumed")
                out_hei = int(self.height_pid(self.fc.state.alt_add.value))
                self.fc.update_realtime_control(vel_z=out_hei)
            else:
                if not paused:
                    paused = True
                    self.height_pid.set_auto_mode(False)
                    self.fc.update_realtime_control(vel_z=0)
                    logger.info("[NAVI] Keep height paused")

    def _navigation_task(self):
        paused = False
        while self.running:
            time.sleep(0.01)
            if self.navigation_flag and self.fc.state.mode.value == self.fc.HOLD_POS_MODE:
                if paused:
                    paused = False
                    self.navi_x_pid.set_auto_mode(True, last_output=0)
                    self.navi_y_pid.set_auto_mode(True, last_output=0)
                    self.navi_yaw_pid.set_auto_mode(True, last_output=0)
                    logger.info("[NAVI] Navigation resumed")
                if self.rs.update_event.wait(1):  # 等待地图更新
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
                    available = self.rs.pose.tracker_confidence == 3
                    if available:
                        # self.fc.send_general_position(x=self.current_x, y=self.current_y)
                        out_x = int(self.navi_x_pid(self.current_x))
                        if out_x is not None:
                            self.fc.update_realtime_control(vel_x=out_x)
                        out_y = int(self.navi_y_pid(self.current_y))
                        if out_y is not None:
                            self.fc.update_realtime_control(vel_y=out_y)
                        out_yaw = int(self.navi_yaw_pid(self.current_yaw))
                        if out_yaw is not None:
                            self.fc.update_realtime_control(yaw=out_yaw)
                    else:
                        logger.warning("[NAVI] Pose from T265 not available")
                        time.sleep(0.1)
                    if 0:  # debug
                        logger.debug(
                            (
                                f"[NAVI] Current pose: {self.current_x}, {self.current_y}, {self.current_yaw}; "
                                f"Output: {out_x}, {out_y}, {out_yaw}"
                            )
                        )
            else:
                if not paused:
                    paused = True
                    self.navi_x_pid.set_auto_mode(False)
                    self.navi_y_pid.set_auto_mode(False)
                    self.navi_yaw_pid.set_auto_mode(False)
                    self.fc.update_realtime_control(vel_x=0, vel_y=0, yaw=0)
                    logger.info("[NAVI] Navigation paused")

    def calibrate_realsense(self):
        """
        根据雷达数据校准T265的位置
        """
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
        logger.debug(f"[NAVI] Calculate T265: z={dx}, x={dy}, y={dz}, yaw={dyaw}")
        self.rs.establish_secondary_origin(force_level=True, z_offset=dx, x_offset=dy, y_offset=dz, yaw_offset=dyaw)

    def navigation_to_waypoint(self, waypoint):
        self.navi_x_pid.setpoint = waypoint[0]
        self.navi_y_pid.setpoint = waypoint[1]
        logger.debug(f"[NAVI] Navigation to waypoint: {waypoint}")

    def set_navigation_speed(self, speed):
        speed = abs(speed)
        self.navi_x_pid.output_limits = (-speed, speed)
        self.navi_y_pid.output_limits = (-speed, speed)
        logger.info(f"[NAVI] Navigation speed set to {speed}")

    def _reached_waypoint(self, pos_thres=15):
        return (
            abs(self.current_pose[0] - self.navi_x_pid.setpoint) < pos_thres
            and abs(self.current_pose[1] - self.navi_y_pid.setpoint) < pos_thres
        )

    def pointing_takeoff(self, point):
        """
        定点起飞
        """
        logger.info(f"[NAVI] Takeoff at {point}")
        self.navigation_flag = False
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
        self.height_pid.setpoint = self.cruise_height
        self.keep_height_flag = True
        time.sleep(2)
        self.navigation_to_waypoint(point)  # 初始化路径点
        self.switch_pid("default")
        time.sleep(0.1)
        self.navigation_flag = True
        self.set_navigation_speed(self.navigation_speed)

    def pointing_landing(self, point):
        """
        定点降落
        """
        logger.info(f"[NAVI] Landing at {point}")
        self.navigation_to_waypoint(point)
        self.wait_for_waypoint()
        self.set_navigation_speed(self.precision_speed)
        self.switch_pid("landing")
        time.sleep(1)
        self.height_pid.setpoint = 60
        time.sleep(1.5)
        self.height_pid.setpoint = 30
        time.sleep(1.5)
        self.height_pid.setpoint = 20
        time.sleep(2)
        self.wait_for_waypoint()
        self.height_pid.setpoint = 0
        # self.fc.land()
        self.fc.wait_for_lock(5)
        self.fc.lock()

    def wait_for_waypoint(self, time_thres=1, pos_thres=15, timeout=30):
        """
        等待到达目标点
        time_thres: 到达目标点后积累的时间
        pos_thres: 到达目标点的距离阈值/cm
        timeout: 超时时间/s
        """
        time_count = 0
        time_start = time.perf_counter()
        while True:
            time.sleep(0.05)
            if self._reached_waypoint(pos_thres):
                time_count += 0.05
            if time_count > time_thres:
                logger.info("[NAVI] Reached waypoint")
                return
            if time.perf_counter() - time_start > timeout:
                logger.warning("[NAVI] Waypoint overtime")
                return

    def wait_for_waypoint_with_avoidance(self, time_thres=1, pos_thres=15, timeout=60):
        time_count = 0
        time_start = time.perf_counter()
        while True:
            time.sleep(0.05)
            if self._reached_waypoint(pos_thres):
                time_count += 0.05
            if time_count > time_thres:
                logger.info("[NAVI] Reached waypoint")
                return
            if time.perf_counter() - time_start > timeout:
                logger.warning("[NAVI] Waypoint overtime")
                return
            self._avoidance_handler()

    def _avoidance_handler(self):
        points = self.radar.map.find_nearest(self._avd_fp_from, self._avd_fp_to, 1, self._avd_fp_dist)
        if len(points) > 0:
            logger.info("[NAVI] Found obstacle")
            waypoint = np.array([self.navi_x_pid.setpoint, self.navi_y_pid.setpoint])
            pos_point = np.array([self.radar.rt_pose[0], self.radar.rt_pose[1]])
            self.navigation_to_waypoint(pos_point)  # 原地停下
            self.height_pid.setpoint = self._avd_height
            self.fc.set_rgb_led(255, 0, 0)
            time.sleep(1)
            self.fc.set_rgb_led(0, 0, 0)
            time.sleep(1)  # 等待高度稳定
            self.keep_height_flag = False
            self.navigation_flag = False
            self.fc.set_flight_mode(self.fc.PROGRAM_MODE)
            self.fc.horizontal_move(self._avd_move, 25, self._avd_deg)
            self.fc.set_rgb_led(255, 255, 0)
            self.fc.wait_for_last_command_done()
            self.fc.set_rgb_led(0, 0, 0)
            self.fc.set_flight_mode(self.fc.HOLD_POS_MODE)
            self.keep_height_flag = True
            self.height_pid.setpoint = self.cruise_height
            time.sleep(1)  # 等待高度稳定
            self.navigation_flag = True
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
