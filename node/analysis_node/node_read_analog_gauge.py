#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg

import math

MAX_PIXEL = 1000

MIN_RADIUS_RATIO = 0.7
MAX_RADIUS_RATIO = 0.95

P1_LOW = 0.0
P1_UP = 0.3
P2_LOW = 0.7
P2_UP = 0.9

#@title 2-1.指示針検出のためのパラメータ
#@markdown 線分と円の中心の距離
P1_LOW = 0.0  #@param {type:"number"}
P1_UP = 0.3   #@param {type:"number"}
#@markdown 線分と円周の距離
P2_LOW = 0.7  #@param {type:"number"}
P2_UP = 0.9   #@param {type:"number"}

#@markdown 2値化関連閾値
THRESH = 100  #@param {type:"number"}
MAX_VALUE = 180 #@param {type:"number"}

#@markdown 指示針検出関連閾値
MIN_LINE_LENGTH = 50 #@param {type:"number"}
MAX_LINE_GAP = 10 #@param {type:"number"}

#@markdown 3点同一線上判定閾値
COLINEAR_THRESHOLD = 1.0  #@param {type:"number"}

def avg_circles(pcircles, pb):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(pb):
        avg_x = avg_x + pcircles[0][i][0]
        avg_y = avg_y + pcircles[0][i][1]
        avg_r = avg_r + pcircles[0][i][2]
    avg_x = int(avg_x/(pb))
    avg_y = int(avg_y/(pb))
    avg_r = int(avg_r/(pb))
    return avg_x, avg_y, avg_r

def dist_2_pts(px1, py1, px2, py2):
    return np.sqrt((px2 - px1)**2 + (py2 - py1)**2)

def get_img_reduce_size(mimg, max_pixel):
    '''get image and reduce size to max_pixel'''
    row, col = mimg.shape[:2]           # get number of rows (height), columns (width)
    if row >= col and row > max_pixel:  # calculate ratio to reduce image
        ratio = max_pixel/row
    elif col >= row and col > max_pixel:
        ratio = max_pixel/col
    else:
        ratio = 1.0
    mimg = cv2.resize(mimg, (0, 0), fx=ratio, fy=ratio)
    mheight, mwidth = mimg.shape[:2]
    mgrey_img = cv2.cvtColor(mimg, cv2.COLOR_BGR2GRAY) # convert to grey image
    return mimg, mgrey_img, mheight, mwidth

def get_circle_and_crop_image(pimg, red_ratio, minrr, maxrr):
    mheight, mwidth = pimg.shape[:2]
    new_height = int(mheight*red_ratio)
    new_width = int(mwidth*red_ratio)
    mx1 = (mwidth - new_width)
    my1 = (mheight - new_height)
    mimg = pimg[my1:new_height, mx1:new_width]
    mgrey_img = cv2.cvtColor(mimg, cv2.COLOR_BGR2GRAY) # convert to grey image
    mgrey_blured_img = cv2.medianBlur(mgrey_img, 5)
    mheight, mwidth = mgrey_blured_img.shape[:2]
    circles = cv2.HoughCircles(mgrey_blured_img, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50,
                              int(mheight*minrr/2), int(mheight*maxrr/2))

    b = circles.shape[1]
    mcircles_img = mimg.copy()
    mcircle_img = mimg.copy()
    for (mx, my, mr) in circles[0, :]:
        cv2.circle(mcircles_img, (int(mx), int(my)), int(mr), (0, 255, 0), 3)
        cv2.circle(mcircles_img, (int(mx), int(my)), 2, (0, 255, 0), 3)
    mx, my, mr = avg_circles(circles, b) # averaging out nearby circles
    cv2.circle(mcircle_img, (mx, my), mr, (0, 255, 0), 3)
    cv2.circle(mcircle_img, (mx, my), 2, (0, 255, 0), 3)
    rect_x = (mx - mr)                  # crop image to circle (x=r, y=r)
    rect_y = (my - mr)
    cropped_img = mimg[rect_y:(rect_y+2*mr), rect_x:(rect_x+2*mr)]
    cropped_circle_img = mcircle_img[rect_y:(rect_y+2*mr), rect_x:(rect_x+2*mr)]
    cropped_grey_img = mgrey_img[rect_y:(rect_y+2*mr), rect_x:(rect_x+2*mr)]
    mheight, mwidth = cropped_circle_img.shape[:2]
    return mr, mr, mr, mcircles_img, cropped_img, cropped_circle_img, \
        cropped_grey_img

def get_pointer(px, py, r, pimg, pgrey_img, p1_b_low, p1_b_up, p2_b_low, p2_b_up):
    mgrey_img = cv2.medianBlur(pgrey_img, 5)
    threshhold_img = cv2.threshold(mgrey_img, THRESH, MAX_VALUE, cv2.THRESH_BINARY_INV)[1]
    lines = cv2.HoughLinesP(image=threshhold_img, rho=3, theta=np.pi / 180, threshold=100,
                           minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    mlines_img = pimg.copy()
    mline_img = pimg.copy()
    for line in lines:                  # create image with lines
        mx1, my1, mx2, my2 = line[0]
        cv2.line(mlines_img, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
    valid, mx1, my1, mx2, my2 = calculate_pointer(px, py, r, lines, p1_b_low, p1_b_up,
                                           p2_b_low, p2_b_up)

    if (valid == False):
        return 0, 0, 0, 0, threshhold_img, mlines_img, None

    cv2.line(mline_img, (mx1, my1), (mx2, my2), (0, 255, 0), 2) # create image w line

    return mx1, my1, mx2, my2, threshhold_img, mlines_img, mline_img

def calculate_pointer(px, py, r, plines, p1_b_low, p1_b_up, p2_b_low, p2_b_up):
    '''calculate the pointer'''
    final_line_list = []
    for i, pline in enumerate(plines):
        for mx1, my1, mx2, my2 in pline:
            diff1 = dist_2_pts(px, py, mx1, my1)  # x, y is center of circle
            diff2 = dist_2_pts(px, py, mx2, my2)  # x, y is center of circle
            if diff1 > diff2:             # set diff1 to be the smaller (closest to center)
                diff1, diff2 = diff2, diff1  # of the two,makes the math easier
            if (p1_b_low*r < diff1 < p1_b_up*r) and \
               (p2_b_low*r < diff2 < p2_b_up*r): # check if in acceptable range
#                final_line_list.append([mx1, my1, mx2, my2]) # add to final list
                if (is_collinear((px, py), (mx1, my1), (mx2, my2))):    # 線分と円の中心が同一線上にあるかどうかを調べる
                    final_line_list.append([mx1, my1, mx2, my2]) # add to final list
    try:
        mx1 = final_line_list[0][0]
        my1 = final_line_list[0][1]
        mx2 = final_line_list[0][2]
        my2 = final_line_list[0][3]
    except IndexError:
        return False, 0, 0, 0, 0

    return True, mx1, my1, mx2, my2

def draw_calibration_circle(img, x, y, r):
    #draw center and circle
    cimg = img.copy()
    cv2.circle(cimg, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(cimg, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    separation = 10.0 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 0.9 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 0.9 * r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(cimg, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(cimg, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0),1,cv2.LINE_AA)

    return cimg

def calculate_angle_and_value(px, py, px1, py1, px2, py2, min_angle, max_angle, min_value, max_value):
    '''calculate the angle and value'''
    dist_pt0 = dist_2_pts(px, py, px1, py1)
    dist_pt1 = dist_2_pts(px, py, px2, py2)
    if dist_pt0 > dist_pt1:
        xlen = px1 - px
        ylen = py - py1
    else:
        xlen = px2 - px
        ylen = py - py2
    if xlen == 0:
        xlen = 0.0000000000000000001
    res = np.arctan(np.divide(float(abs(ylen)), float(abs(xlen)))) # arc-tan
    res = np.rad2deg(res)

    if xlen > 0 and ylen > 0:  #in quadrant I
        final_angle = 270 - res
    if xlen < 0 and ylen > 0:  #in quadrant II
        final_angle = 90 - res
    if xlen < 0 and ylen < 0:  #in quadrant III
        final_angle = 90 - res
    if xlen > 0 and ylen < 0:  #in quadrant IV
        final_angle = 270 - res

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    value = new_value

    return final_angle, value

# 3つの点が同一線上にあるかどうかを返す
def is_collinear(a, b, c):
    def equal_with_tolerance(x, y):
        return abs(x - y) < COLINEAR_THRESHOLD

    ab = distance(a, b)
    ac = distance(a, c)
    bc = distance(b, c)

    return equal_with_tolerance(ab + ac, bc) or equal_with_tolerance(ab + bc, ac) or equal_with_tolerance(ac + bc, ab)

# ユークリッド距離の計算
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# 3点の座標の角度を返す
def calculate_angle(x1, y1, x2, y2, x3, y3):
    # ベクトルABの成分を計算
    ab_x = x2 - x1
    ab_y = y2 - y1

    # ベクトルBCの成分を計算
    bc_x = x3 - x2
    bc_y = y3 - y2

    # ベクトルABとBCの内積を計算
    dot_product = ab_x * bc_x + ab_y * bc_y

    # ベクトルABとBCの大きさを計算
    magnitude_ab = math.sqrt(ab_x ** 2 + ab_y ** 2)
    magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)

    # 角度を計算
    cos_theta = dot_product / (magnitude_ab * magnitude_bc)
    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)
    return theta_deg

def image_process(image, min_angle, max_angle, min_value, max_value):
    image, grey_img, height, width = get_img_reduce_size(image, MAX_PIXEL)
    try:
        x, y, r, circles_img, image, circle_img, grey_img = get_circle_and_crop_image(image, 1, MIN_RADIUS_RATIO, MAX_RADIUS_RATIO)
        calib_img = draw_calibration_circle(circle_img, x, y, r)
    except:
        return image, None

    #@title 2-2.指示針検出・読取結果表示
    try:
        x1, y1, x2, y2, threshhold_image, lines_img, line_img = get_pointer(x, y, r, circle_img, grey_img, P1_LOW, P1_UP, P2_LOW, P2_UP)
    except:
        return calib_img, None

    if (line_img is None):
        return calib_img, None
    
    final_img = draw_calibration_circle(line_img, x, y, r)
    try:
        final_angle, value = calculate_angle_and_value(x, y, x1, y1, x2, y2, min_angle, max_angle, min_value, max_value)
    except:
        return final_img, None

    return final_img, value

class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Analog Gauge Reader 1'
    node_tag = 'AnalogGaugeReader1'

    _min_val = 1
    _max_val = 128

    _opencv_setting_dict = None

    def __init__(self):
        pass

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        # タグ名
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_INT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_INT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        tag_node_input04_name = tag_node_name + ':' + self.TYPE_INT + ':Input04'
        tag_node_input04_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        tag_node_input05_name = tag_node_name + ':' + self.TYPE_INT + ':Input05'
        tag_node_input05_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input05Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        tag_node_output03_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output03'
        tag_node_output03_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output03Value'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

        # 初期化用黒画像
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

        # テクスチャ登録
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # 入力端子
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image',
                )
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # 最小角度
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=tag_node_input02_value_name,
                    label="Min angle",
                    width=small_window_w - 80,
                    default_value=50,
                    min_value=0,
                    max_value=360,
                    callback=None,
                )
            # 最大角度
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=tag_node_input03_value_name,
                    label="Max angle",
                    width=small_window_w - 80,
                    default_value=310,
                    min_value=0,
                    max_value=360,
                    callback=None,
                )
            # 最小角度の時の値
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=tag_node_input04_value_name,
                    label="Min value",
                    width=small_window_w - 80,
                    default_value=10,
                    min_value=self._min_val,
                    max_value=self._max_val,
                    callback=None,
                )
            # 最大角度の時の値
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=tag_node_input05_value_name,
                    label="Max value",
                    width=small_window_w - 80,
                    default_value=230,
                    min_value=self._min_val,
                    max_value=self._max_val,
                    callback=None,
                )
            # 処理時間
            if use_pref_counter:
                with dpg.node_attribute(
                        tag=tag_node_output02_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=tag_node_output02_value_name,
                        default_value='elapsed time(ms)',
                    )
            # 指示値
            with dpg.node_attribute(
                    tag=tag_node_output03_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=tag_node_output03_value_name,
                    default_value='0',
                )

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_INT + ':Input05Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        output_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output03Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        value = 0

        # 接続情報確認
        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = int(dpg_get_value(source_tag))
                input_value = max([self._min_val, input_value])
                input_value = min([self._max_val, input_value])
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                connection_info_src = ':'.join(connection_info_src)

        # 画像取得
        frame = node_image_dict.get(connection_info_src, None)

        # 角度
        min_angle = int(dpg_get_value(input_value02_tag))
        max_angle = int(dpg_get_value(input_value03_tag))

        # 値
        min_value = int(dpg_get_value(input_value04_tag))
        max_value = int(dpg_get_value(input_value05_tag))

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frame is not None:
            frame, value = image_process(frame, min_angle, max_angle, min_value, max_value)
            value = round(value, 3)

        # 計測終了
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')
            dpg_set_value(output_value03_tag,
                          value)

        # 描画
        if frame is not None:
            texture = convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, str(value)

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        min_angle = dpg_get_value(input_value02_tag)
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        max_angle = dpg_get_value(input_value03_tag)
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        min_value = dpg_get_value(input_value04_tag)
        input_value05_tag = tag_node_name + ':' + self.TYPE_INT + ':Input05Value'
        max_value = dpg_get_value(input_value05_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = min_angle
        setting_dict[input_value03_tag] = max_angle
        setting_dict[input_value04_tag] = min_value
        setting_dict[input_value05_tag] = max_value

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_INT + ':Input05Value'

        min_angle = int(setting_dict[input_value02_tag])
        max_angle = int(setting_dict[input_value03_tag])
        min_value = int(setting_dict[input_value04_tag])
        max_value = int(setting_dict[input_value05_tag])

        dpg_set_value(input_value02_tag, min_angle)
        dpg_set_value(input_value03_tag, max_angle)
        dpg_set_value(input_value04_tag, min_value)
        dpg_set_value(input_value05_tag, max_value)

