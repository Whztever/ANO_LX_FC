import threading
import time
import traceback

import cv2
import numpy as np
import serial
from FlightController.Components.LDRadar_Resolver import (
    Map_Circle,
    Point_2D,
    Radar_Package,
    resolve_radar_data,
)
from FlightController.Solutions.Radar import radar_resolve_rt_pose
from loguru import logger

DEBUG_SAVE_IMAGE = False


class LD_Radar(object):
    def __init__(self):
        self.map = Map_Circle()
        self.running = False
        self._thread_list = []
        self._package = Radar_Package()
        self._serial = None
        self._update_callback = None
        self.subtask_event = threading.Event()
        self.subtask_skip = 4
        # 位姿估计
        self.rt_pose_update_event = threading.Event()
        self.rt_pose = [0, 0, 0]
        self._rtpose_flag = False
        self._rt_pose_inited = [False, False, False]
        # 解析函数
        self._map_funcs = []
        self.map_func_update_times = []
        self.map_func_results = []

    def start(self, com_port, radar_type: str = "LD08", update_callback=None, subtask_skip=4):
        """
        开始监听雷达数据
        radar_type: LD08 or LD06
        update_callback: 回调函数, 次更新雷达数据时调用
        subtask_skip: 多少次雷达数据更新后, 进行一次子任务
        """
        if self.running:
            self.stop()
        if radar_type == "LD08":
            baudrate = 115200
        elif radar_type == "LD06":
            baudrate = 230400
        else:
            raise ValueError("Unknown radar type")
        self._serial = serial.Serial(com_port, baudrate=baudrate)
        self._update_callback = update_callback
        self.subtask_event.clear()
        self.subtask_skip = subtask_skip
        self.running = True
        thread = threading.Thread(target=self._read_serial_task)
        thread.daemon = True
        thread.start()
        self._thread_list.append(thread)
        logger.info("[RADAR] Listenning thread started")
        thread = threading.Thread(target=self._map_resolve_task)
        thread.daemon = True
        thread.start()
        self._thread_list.append(thread)
        logger.info("[RADAR] Map resolve thread started")
        self.start_time = time.perf_counter()

    def stop(self, joined=False):
        """
        停止监听雷达数据
        """
        self.running = False
        if joined:
            for thread in self._thread_list:
                thread.join()
        if self._serial != None:
            self._serial.close()
        logger.info("[RADAR] Stopped all threads")

    def _read_serial_task(self):
        reading_flag = False
        start_bit = b"\x54\x2C"
        package_length = 45
        read_buffer = bytes()
        wait_buffer = bytes()
        count = 0
        while self.running:
            try:
                if self._serial.in_waiting > 0:
                    if not reading_flag:  # 等待包头
                        wait_buffer += self._serial.read(1)
                        if len(wait_buffer) >= 2:
                            if wait_buffer[-2:] == start_bit:
                                reading_flag = True
                                wait_buffer = bytes()
                                read_buffer = start_bit
                    else:  # 读取数据
                        read_buffer += self._serial.read(package_length)
                        reading_flag = False
                        resolve_radar_data(read_buffer, self._package)
                        self.map.update(self._package)
                        if self._update_callback is not None:
                            self._update_callback(self._package)
                        count += 1
                        if count >= self.subtask_skip:
                            count = 0
                            self.subtask_event.set()
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"[RADAR] Listenning thread error: {traceback.format_exc()}")
                time.sleep(0.5)

    def _map_resolve_task(self):
        while self.running:
            try:
                if self.subtask_event.wait(1):
                    self.subtask_event.clear()
                    if self._rtpose_flag:
                        # img = self.map.output_cloud(
                        #     size=int(self._rtpose_size),
                        #     scale=0.1 * self._rtpose_scale_ratio,
                        # )
                        # x, y, yaw = radar_resolve_rt_pose(img)
                        img = self.map.output_polyline_cloud(
                            size=int(self._rtpose_size),
                            scale=0.1 * self._rtpose_scale_ratio,
                            thickness=1,
                            draw_outside=False,
                        )
                        if DEBUG_SAVE_IMAGE:
                            cv2.imwrite("radar_cloud.png", img)
                        x, y, yaw = radar_resolve_rt_pose(img, skip_er=True, skip_di=True)
                        if x is not None:
                            if self._rt_pose_inited[0]:
                                self.rt_pose[0] += (
                                    x / self._rtpose_scale_ratio - self.rt_pose[0]
                                ) * self._rtpose_low_pass_ratio
                            else:
                                self.rt_pose[0] = x / self._rtpose_scale_ratio
                                self._rt_pose_inited[0] = True
                        if y is not None:
                            if self._rt_pose_inited[1]:
                                self.rt_pose[1] += (
                                    y / self._rtpose_scale_ratio - self.rt_pose[1]
                                ) * self._rtpose_low_pass_ratio
                            else:
                                self.rt_pose[1] = y / self._rtpose_scale_ratio
                                self._rt_pose_inited[1] = True
                        if yaw is not None:
                            if self._rt_pose_inited[2]:
                                self.rt_pose[2] += (yaw - self.rt_pose[2]) * self._rtpose_low_pass_ratio
                            else:
                                self.rt_pose[2] = yaw
                                self._rt_pose_inited[2] = True
                        self.rt_pose_update_event.set()
                    for i in range(len(self._map_funcs)):
                        if self._map_funcs[i]:
                            func, args, kwargs = self._map_funcs[i]
                            result = func(self.map, *args, **kwargs)
                            if result:
                                self.map_func_results[i] = result
                                self.map_func_update_times[i] = time.perf_counter()
                else:
                    logger.warning("[RADAR] Map resolve thread wait timeout")
            except Exception as e:
                import traceback

                logger.error(f"[RADAR] Map resolve thread error: {traceback.format_exc()}")
                time.sleep(0.5)

    def _init_radar_map(self):
        self._radar_map_img = np.zeros((600, 600, 3), dtype=np.uint8)
        a = np.sqrt(2) * 600
        b = (a - 600) / 2
        c = a - b
        b = int(b / np.sqrt(2))
        c = int(c / np.sqrt(2))
        cv2.line(self._radar_map_img, (b, b), (c, c), (255, 0, 0), 1)
        cv2.line(self._radar_map_img, (c, b), (b, c), (255, 0, 0), 1)
        cv2.line(self._radar_map_img, (300, 0), (300, 600), (255, 0, 0), 1)
        cv2.line(self._radar_map_img, (0, 300), (600, 300), (255, 0, 0), 1)
        cv2.circle(self._radar_map_img, (300, 300), 100, (255, 0, 0), 1)
        cv2.circle(self._radar_map_img, (300, 300), 200, (255, 0, 0), 1)
        cv2.circle(self._radar_map_img, (300, 300), 300, (255, 0, 0), 1)
        self.__radar_map_img_scale = 1
        self.__radar_map_info_angle = 0
        cv2.namedWindow("Radar Map", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(
            "Radar Map",
            lambda *args, **kwargs: self._radar_map_on_mouse(*args, **kwargs),
        )

    def _radar_map_on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.__radar_map_img_scale *= 1.1
            else:
                self.__radar_map_img_scale *= 0.9
            self.__radar_map_img_scale = min(max(0.001, self.__radar_map_img_scale), 2)
        elif event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            self.__radar_map_info_angle = (90 - np.arctan2(300 - y, x - 300) * 180 / np.pi) % 360

    def show_radar_map(self, add_p_func=None, *args, **kwargs):
        """
        显示雷达地图(调试用, 高占用且阻塞)
        """
        self._init_radar_map()
        while True:
            img_ = self._radar_map_img.copy()
            cv2.putText(
                img_,
                f"{100/self.__radar_map_img_scale:.0f}",
                (300, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
            )
            cv2.putText(
                img_,
                f"{200/self.__radar_map_img_scale:.0f}",
                (300, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
            )
            cv2.putText(
                img_,
                f"{300/self.__radar_map_img_scale:.0f}",
                (300, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
            )
            if add_p_func is not None:
                add_p = add_p_func(*args, **kwargs)
                for i, p in enumerate(add_p):
                    cv2.putText(
                        img_,
                        f"AP-{i}: {p}",
                        (10, 520 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                    )

            else:
                add_p = []

            cv2.putText(
                img_,
                f"Angle: {self.__radar_map_info_angle:.1f} (idx={round((self.__radar_map_info_angle % 360) * self.map.ACC)})",
                (10, 540),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
            )
            cv2.putText(
                img_,
                f"Distance: {self.map.get_distance(self.__radar_map_info_angle)}",
                (10, 560),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
            )
            point = self.map.get_point(self.__radar_map_info_angle)
            if point:
                xy = point.to_xy()
                cv2.putText(
                    img_,
                    f"Position: ( {xy[0]:.2f} , {xy[1]:.2f} )",
                    (10, 580),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                )
                add_p = [point] + add_p
                pos = point.to_cv_xy() * self.__radar_map_img_scale + np.array([300, 300])
                cv2.line(img_, (300, 300), (int(pos[0]), int(pos[1])), (255, 255, 0), 1)

            self.map.draw_on_cv_image(img_, scale=self.__radar_map_img_scale, add_points=add_p)
            cv2.putText(
                img_,
                f"RPM={self.map.rotation_spd:05.2f} PPS={self.map.update_count/(time.perf_counter()-self.start_time):05.2f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
            )
            cv2.putText(
                img_,
                f"AVAIL={self.map.avail_points}/{self.map.total_points} CNT={self.map.update_count} ",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
            )
            cv2.imshow("Radar Map", img_)
            key = cv2.waitKey(int(1000 / 50))
            if key == ord("q"):
                cv2.destroyWindow("Radar Map")
                break
            elif key == ord("w"):
                self.__radar_map_img_scale *= 1.1
            elif key == ord("s"):
                self.__radar_map_img_scale *= 0.9
            elif key == ord("a"):
                t0 = time.perf_counter()
                out = self.map.output_polyline_cloud(scale=self.__radar_map_img_scale, size=800, draw_outside=False)
                t1 = time.perf_counter()
                print(f"output_polyline_cloud: {t1 - t0:.9f}s")
                cv2.imshow("Cloud(polyline)", out)
                t0 = time.perf_counter()
                out = self.map.output_cloud(scale=self.__radar_map_img_scale, size=800)
                t1 = time.perf_counter()
                print(f"output_cloud: {t1 - t0:.9f}s")
                cv2.imshow("Cloud", out)
                # cv2.imwrite(f"radar_map.png", out)

    def register_map_func(self, func, *args, **kwargs) -> int:
        """
        注册雷达地图解析函数

        func应为self.map包含的方法,所有附加参数将会传递给该func
        从列表map_func_results中获取结果,
        列表map_func_update_times储存了上一次该函数返回非空结果的时间,用于超时判断

        return: func_id
        """
        self._map_funcs.append((func, args, kwargs))
        self.map_func_results.append(None)
        self.map_func_update_times.append(0)
        return len(self._map_funcs) - 1

    def unregister_map_func(self, func_id: int):
        """
        注销雷达地图解析函数
        """
        self._map_funcs[func_id] = None
        self.map_func_results[func_id] = None
        self.map_func_update_times[func_id] = 0

    def update_map_func_args(self, func_id: int, *args, **kwargs):
        """
        更新雷达地图解析函数参数
        """
        self._map_funcs[func_id][1] = args
        self._map_funcs[func_id][2] = kwargs

    def start_resolve_pose(self, size: int = 1000, scale_ratio: float = 1, low_pass_ratio: float = 0.5):
        """
        开始使用点云图解算位姿
        size: 解算范围(长宽为size的正方形)
        scale_ratio: 降采样比例, 降低精度节省计算资源
        low_pass_ratio: 低通滤波比例
        """
        self._rtpose_flag = True
        self._rtpose_size = size
        self._rtpose_scale_ratio = scale_ratio
        self._rtpose_low_pass_ratio = low_pass_ratio
        self.rt_pose = [0, 0, 0]
        self._rt_pose_inited = [False, False, False]

    def stop_resolve_pose(self):
        """
        停止使用点云图解算位姿
        """
        self._rtpose_flag = False
        self.rt_pose = [0, 0, 0]
        self._rt_pose_inited = [False, False, False]
        self.rt_pose_update_event.clear()

    def update_resolve_pose_args(self, size=None, ratio=None, low_pass_ratio=None):
        """
        更新位姿参数
        size: 解算范围(长宽为size的正方形)
        scale_ratio: 降采样比例, 降低精度节省计算资源
        low_pass_ratio: 低通滤波比例
        """
        self._rtpose_size = size if size is not None else self._rtpose_size
        self._rtpose_scale_ratio = ratio if ratio is not None else self._rtpose_scale_ratio
        self._rtpose_low_pass_ratio = low_pass_ratio if low_pass_ratio is not None else self._rtpose_low_pass_ratio
