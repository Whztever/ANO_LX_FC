from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

from ..Components.LDRadar_Resolver import Point_2D

############参数设置##############
KERNAL_DI = 9  # 膨胀核大小
KERNAL_ER = 5  # 腐蚀核大小
HOUGH_THRESHOLD = 80
MIN_LINE_LENGTH = 60
# KERNAL_DI = 9  # 膨胀核大小
# KERNAL_ER = 5  # 腐蚀核大小
# HOUGH_THRESHOLD = 50
# MIN_LINE_LENGTH = 60
MAX_LINE_GAP = 200
#################################
kernel_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNAL_DI, KERNAL_DI))
kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNAL_ER, KERNAL_ER))

_and = np.logical_and
_or = np.logical_or
_not = np.logical_not


def radar_resolve_rt_pose(
    img, debug=False, skip_di=False, skip_er=False
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    从雷达点云图像中解析出中点位置
    img: 雷达点云图像(灰度图)
    debug: 显示解析结果
    skip_di: 是否跳过膨胀
    skip_er: 是否跳过腐蚀
    return: 位姿(x,y,yaw)
    """
    if not skip_di:
        img = cv2.dilate(img, kernel_di)  # 膨胀
    if not skip_er:
        img = cv2.erode(img, kernel_er)  # 腐蚀
    lines = cv2.HoughLinesP(
        img,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )
    if lines is None:
        return None, None, None
    x0, y0 = img.shape[0] // 2, img.shape[1] // 2
    if debug:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    x_out = None
    y_out = None
    yaw_out_1 = None
    yaw_out_2 = None

    # mathod 1: 我也忘了这堆条件怎么想的了, 要配合选取最近线的条件
    # select_right = _and(
    #     _and(lines[:, :, 0] > x0, lines[:, :, 2] > x0),
    #     _or(_and(lines[:, :, 1] > y0, lines[:, :, 3] < y0), _and(lines[:, :, 1] < y0, lines[:, :, 3] > y0)),
    # )
    # select_back = _and(
    #     _and(lines[:, :, 1] > y0, lines[:, :, 3] > y0),
    #     _or(_and(lines[:, :, 0] > x0, lines[:, :, 2] < x0), _and(lines[:, :, 0] < x0, lines[:, :, 2] > x0)),
    # )

    # mathod 2: 中心点画个X, 根据中点在X四个区域中的哪个判断是哪一侧的直线, 选取角度差最小的直线
    midpoints = (lines[:, :, :2] + lines[:, :, 2:]) / 2
    # 原点指向中点的向量角度
    degs = np.degrees(np.arctan2(midpoints[:, :, 1] - y0, midpoints[:, :, 0] - x0))
    select_right = _and(degs > -45, degs < 45)
    select_back = _and(degs > 45, degs < 135)

    right_lines = lines[select_right]
    back_lines = lines[select_back]

    if right_lines.shape[0] != 0:
        dists, angles = get_point_line_distance_np([x0, y0], right_lines)
        # line_index = np.argmin(right_dists) # 选取距离最近的直线
        line_index = np.argmin(np.abs(angles))  # 选取角度最小的直线
        y_out = dists[line_index]
        yaw_out_1 = -angles[line_index]

    if back_lines.shape[0] != 0:
        dists, angles = get_point_line_distance_np([x0, y0], back_lines)
        # line_index = np.argmin(back_dists)
        line_index = np.argmin(np.abs(angles - 90))
        x_out = dists[line_index]
        yaw_out_2 = -angles[line_index] + 90

    if yaw_out_1 is not None and yaw_out_2 is not None:
        yaw_out = (yaw_out_1 + yaw_out_2) / 2
    elif yaw_out_1 is not None:
        yaw_out = yaw_out_1
    elif yaw_out_2 is not None:
        yaw_out = yaw_out_2
    else:
        yaw_out = None

    if debug:
        for x1, y1, x2, y2 in right_lines:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
        for x1, y1, x2, y2 in back_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        for x1, y1, x2, y2 in lines[_not(_or(select_right, select_back))]:
            cv2.line(img, (x1, y1), (x2, y2), (180, 0, 180), 1)
        x_ = x_out if x_out else -1
        y_ = y_out if y_out else -1
        yaw_ = yaw_out if yaw_out else 0
        target = Point_2D(-yaw_, 50).to_cv_xy() + np.array([x0, y0])
        cv2.line(img, (x0, y0), (int(target[0]), int(target[1])), (0, 0, 255), 1)
        if yaw_out_1 is not None:
            target = Point_2D(-yaw_out_1 + 90, y_).to_cv_xy() + np.array([x0, y0])
            cv2.line(img, (x0, y0), (int(target[0]), int(target[1])), (200, 200, 0), 1)
        if yaw_out_2 is not None:
            target = Point_2D(-yaw_out_2 + 180, x_).to_cv_xy() + np.array([x0, y0])
            cv2.line(img, (x0, y0), (int(target[0]), int(target[1])), (0, 200, 200), 1)
        cv2.putText(
            img,
            f"({x_:.1f}, {y_:.1f}, {yaw_:.1f})",
            (x0, y0 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
        )
        cv2.imshow("Map Resolve", img)
    return x_out, y_out, yaw_out


def get_point_line_distance_np(point, lines) -> tuple[np.ndarray, np.ndarray]:
    """
    分别计算一个点到各条线的距离
    point: 目标点 [x,y]
    lines: 线的两个端点 [[x1,y1,x2,y2],...]
    return: 距离, 角度(-90~90)
    """
    # point = np.asarray(point)
    # lines = np.asarray(lines)
    x1, y1, x2, y2 = lines.T

    # 计算线段长度
    line_lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 计算点到线段的投影长度
    projection_lengths = ((point[0] - x1) * (x2 - x1) + (point[1] - y1) * (y2 - y1)) / line_lengths

    # 计算投影点坐标
    px = x1 + projection_lengths * (x2 - x1) / line_lengths
    py = y1 + projection_lengths * (y2 - y1) / line_lengths

    # 计算点到投影点的距离
    distances = np.sqrt((point[0] - px) ** 2 + (point[1] - py) ** 2)

    # 计算角度
    angles = np.degrees(np.arctan2(py - point[1], px - point[0]))

    return distances, angles
