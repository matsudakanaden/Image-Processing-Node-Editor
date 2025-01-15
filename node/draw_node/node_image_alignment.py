#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import re
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info

def image_process(image1, image2, ratio, k_val, thres):
    akaze = cv2.AKAZE_create()
    img1_kp, img1_des = akaze.detectAndCompute(image1, None)
    img2_kp, img2_des = akaze.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img1_des, img2_des, k = k_val)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])

    # 適切なキーポイントを選択
    img2_matched_kpts = np.float32(
        [img1_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32(
        [img2_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ホモグラフィを計算
    H, status = cv2.findHomography(img2_matched_kpts, sensed_matched_kpts, cv2.RANSAC, thres)

    # 画像を変換
    warped_image = cv2.warpPerspective(image1, H, (image1.shape[1], image1.shape[0]))

    return warped_image

def create_image_dict(
    slot_num,
    connection_info_src_dict,
    node_image_dict,
    node_result_dict,
    image_node_name,
    resize_width,
    resize_height,
    draw_info_on_result,
):
    frame_exist_flag = False

    # 初期化用黒画像
    black_image = np.zeros((resize_height, resize_width, 3)).astype(np.uint8)

    frame_dict = {}
    for index in range(slot_num - 1, -1, -1):
        node_id_name = connection_info_src_dict.get(index, None)
        frame = copy.deepcopy(node_image_dict.get(node_id_name, None))
        if frame is not None:
            if draw_info_on_result:
                node_result = node_result_dict[node_id_name]
                image_node_name = node_id_name.split(':')[1]
                frame = draw_info(image_node_name, node_result, frame)
            resize_frame = cv2.resize(frame, (resize_width, resize_height))
            frame_dict[slot_num - index - 1] = copy.deepcopy(resize_frame)

            frame_exist_flag = True
        else:
            frame_dict[slot_num - index - 1] = copy.deepcopy(black_image)

    display_num_list = [1, 2, 4, 4, 6, 6, 9, 9, 9]
    for index in range(display_num_list[slot_num - 1]):
        if frame_dict.get(index, None) is None:
            frame_dict[index] = copy.deepcopy(black_image)

    if not frame_exist_flag:
        frame_dict = None

    return frame_dict


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Image Alignment'
    node_tag = 'ImageAlignment'

    _max_slot_number = 2
    _slot_id = {}

    _k_default = 2
    _thres_default = 5.0
    _ratio_default = 0.75

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
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        tag_node_input04_name = tag_node_name + ':' + self.TYPE_INT + ':Input04'
        tag_node_input04_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        tag_node_input05_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05'
        tag_node_input05_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

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

        # スロットナンバー保持用Dict
        if tag_node_name not in self._slot_id:
            self._slot_id[tag_node_name] = 1

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
                    default_value='Target image',
                )
            # 入力端子
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input02_value_name,
                    default_value='Reference image',
                )
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # 距離閾値
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=tag_node_input03_value_name,
                    label="ratio",
                    width=small_window_w - 80,
                    default_value=self._ratio_default,
                    callback=None,
                )
            # kパラメータ
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=tag_node_input04_value_name,
                    label="k",
                    width=small_window_w - 80,
                    default_value=self._k_default,
                    callback=None,
                )
            # ホモグラフィ閾値
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=tag_node_input05_value_name,
                    label="threshold",
                    width=small_window_w - 80,
                    default_value=self._thres_default,
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

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        draw_info_on_result = self._opencv_setting_dict['draw_info_on_result']

        # 接続情報確認
        frame = None
        frame1 = None
        frame2 = None
        node_name_dict = {}
        connection_info_src = ''
        connection_info_src_dict = {}
        for connection_info in connection_list:

            # タグ名からスロットナンバー取得
            slot_number = re.sub(r'\D', '', connection_info[1].split(':')[-1])
            if slot_number == '':
                continue
            slot_number = int(slot_number) - 1
            connection_type = connection_info[0].split(':')[2]
            connection_tag = connection_info[1].split(':')[3]
            if connection_type == self.TYPE_FLOAT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = round(float(dpg_get_value(source_tag)),3)
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = int(dpg_get_value(source_tag))
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection_info_src = connection_info[0]
                connection_info_src = connection_info_src.split(':')[:2]
                node_name = connection_info_src[1]
                connection_info_src = ':'.join(connection_info_src)
                node_name_dict[slot_number] = node_name
                connection_info_src_dict[slot_number] = connection_info_src

        # 画像取得

        if len(connection_info_src_dict) == 1:
            connected_first_slot_no = (next(iter(connection_info_src_dict)))
            frame1 = node_image_dict.get(connection_info_src_dict[connected_first_slot_no])
            frame = frame1
        if len(connection_info_src_dict) == 2:
            frame1 = node_image_dict.get(connection_info_src_dict[0])  
            frame2 = node_image_dict.get(connection_info_src_dict[1])
            frame = frame1

        # パラメータ
        ratio = float(dpg_get_value(input_value03_tag))
        k = int(dpg_get_value(input_value04_tag))
        thres = float(dpg_get_value(input_value05_tag))

        # 計測開始
        if frame is not None and use_pref_counter:
            start_time = time.perf_counter()
        
        if len(connection_info_src_dict) == 2:
            if frame1 is not None and frame2 is not None:
                frame = image_process(frame1, frame2, ratio, k, thres)

        # 計測終了
        if frame is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')

        # 描画
        if frame is not None:
            texture = convert_cv_to_dpg(
                frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return frame, None

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'

        ratio = dpg_get_value(input_value03_tag)
        k = dpg_get_value(input_value04_tag)
        thres = dpg_get_value(input_value05_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value03_tag] = ratio
        setting_dict[input_value04_tag] = k
        setting_dict[input_value05_tag] = thres

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input05Value'

        ratio = float(setting_dict[input_value03_tag])
        k = int(setting_dict[input_value04_tag])
        thres = float(setting_dict[input_value05_tag])

        dpg_set_value(input_value03_tag, ratio)
        dpg_set_value(input_value04_tag, k)
        dpg_set_value(input_value05_tag, thres)
