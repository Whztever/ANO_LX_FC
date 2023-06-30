"""
使用雷达作为位置闭环的任务模板
"""
import threading
import time
from typing import List

import cv2
import numpy as np
from configManager import ConfigManager
from FlightController import FC_Client, FC_Controller
from FlightController.Components import LD_Radar
from FlightController.Components.RealSense import T265
from FlightController.Solutions.Navigation import Navigation
from FlightController.Solutions.Vision import (
    change_cam_resolution,
    set_cam_autowb,
    vision_debug,
)
from FlightController.Solutions.Vision_Net import FastestDetOnnx
from hmi import HMI
from loguru import logger


def deg_360_180(deg):
    if deg > 180:
        deg = deg - 360
    return deg


# cfg = ConfigManager()
#
# # 基地点
# BASE_POINT = cfg.get_array("point-0")
# logger.info(f"[MISSION] Loaded base point: {BASE_POINT}")
# # 降落点
# landing_point = BASE_POINT
# # 任务坐标
# NULL_PT = np.array([np.NaN, np.NaN])
# POINT = lambda x: cfg.get_array(f"point-{x}")
# POINTS_ARR = np.array(
#     [
#         [POINT(1), NULL_PT, POINT(11), NULL_PT, POINT(5)],
#         [NULL_PT, POINT(8), NULL_PT, POINT(3), NULL_PT],
#         [POINT(9), NULL_PT, POINT(2), NULL_PT, POINT(12)],
#         [NULL_PT, POINT(7), NULL_PT, POINT(6), NULL_PT],
#         [NULL_PT, NULL_PT, POINT(10), NULL_PT, POINT(4)],
#     ]
# )
# logger.info(f"[MISSION] Loaded points: {POINTS_ARR}")
# TARGET_POINTS = [cfg.get_array("target-x")]
# logger.info(f"[MISSION] Loaded target points: {TARGET_POINTS}")
BASE_POINT = np.array([0, 0])
TARGET_POINTS = np.array([[200, 0], [200, -300]])
LANDING_POINT = np.array([200, -300])


class Mission(object):
    def __init__(self, fc: FC_Controller, radar: LD_Radar, camera: cv2.VideoCapture, hmi: HMI, rs: T265):
        self.fc = fc
        self.radar = radar
        self.cam = camera
        self.hmi = hmi
        self.rs = rs
        self.inital_yaw = self.fc.state.yaw.value
        # self.fd = FastestDetOnnx(drawOutput=True)  # 初始化神经网络
        self.navi = Navigation(fc, radar, rs)

    def stop(self):
        self.navi.stop()
        logger.info("[MISSION] Mission stopped")

    def run(self):
        fc = self.fc
        radar = self.radar
        cam = self.cam
        ############### 参数 #################
        self.camera_down_pwm = 32.5
        self.camera_up_pwm = 72
        self.navigation_speed = 35  # 导航速度
        self.precision_speed = 25  # 精确速度
        self.cruise_height = 140  # 巡航高度
        self.goods_height = 80  # 处理物品高度
        self.vertical_speed = 20  # 垂直速度
        self.navi.pid_tunings = {
            "default": (0.35, 0.0, 0.08),  # 导航
            "delivery": (0.4, 0.05, 0.16),  # 配送
            "landing": (0.4, 0.05, 0.16),  # 降落
        }  # PID参数 (仅导航XY使用)
        ################ 启动线程 ################
        fc.set_flight_mode(fc.PROGRAM_MODE)
        self.navi.set_navigation_speed(self.navigation_speed)
        self.navi.set_vertical_speed(self.vertical_speed)
        self.navi.start()  # 启动导航线程
        logger.info("[MISSION] Navigation started")
        ################ 初始化 ################
        fc.set_action_log(False)
        # change_cam_resolution(cam, 800, 600)
        # set_cam_autowb(cam, False)  # 关闭自动白平衡
        # for _ in range(10):
        #     cam.read()
        # fc.set_PWM_output(0, self.camera_up_pwm)
        # self.recognize_targets()
        # for i in range(6):
        #     sleep(0.25)
        #     fc.set_rgb_led(255, 0, 0)  # 起飞前警告
        #     sleep(0.25)
        #     fc.set_rgb_led(0, 0, 0)
        # fc.set_PWM_output(0, self.camera_down_pwm)
        # fc.set_digital_output(2, True)  # 激光笔开启
        fc.set_action_log(True)
        self.navi.switch_pid("default")
        ################ 初始化完成 ################
        logger.info("[MISSION] Mission-1 Started")
        self.navi.pointing_takeoff(BASE_POINT, self.cruise_height)
        ################ 开始任务 ################
        for target_point in TARGET_POINTS:
            self.navi.navigation_to_waypoint(target_point)
            self.navi.wait_for_waypoint()
        ################ 降落 ################
        logger.info("[MISSION] Go to landing point")
        self.navi.navigation_to_waypoint(LANDING_POINT)
        self.navi.wait_for_waypoint()
        self.navi.pointing_landing(LANDING_POINT)
        logger.info("[MISSION] Misson-1 Finished")


if __name__ == "__main__":
    logger.warning("DEBUG MODE!!")
    t265 = T265()
    t265.start()
    # t265.hardware_reset()
    cam = None
    hmi = None
    # fc = FC_Controller()
    # fc.start_listen_serial("/dev/ttyS6", print_state=True)
    fc = FC_Client()
    fc.connect()
    fc.start_sync_state(False)
    fc.wait_for_connection()
    radar = LD_Radar()
    radar.start("/dev/ttyUSB0", "LD06")

    mission = Mission(fc, radar, cam, hmi, t265)  # type: ignore
    logger.warning("Press Enter to start mission")
    input()
    fc._print_state_flag = False
    try:
        mission.run()
    except Exception as e:
        import traceback

        logger.error(f"[MANAGER] Mission Failed: {traceback.format_exc()}")
    finally:
        mission.stop()
        if fc.state.unlock.value:
            logger.warning("[MANAGER] Auto Landing")
            fc.set_flight_mode(fc.PROGRAM_MODE)
            fc.stablize()
            fc.land()
            ret = fc.wait_for_lock()
            if not ret:
                fc.lock()
    logger.info("[MANAGER] Mission finished")
