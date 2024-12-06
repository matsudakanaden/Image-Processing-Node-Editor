#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg

def calculate_similarity(img1, img2, ratio, k_val):
    akaze = cv2.AKAZE_create()
    img1_kp, img1_des = akaze.detectAndCompute(img1, None)
    img2_kp, img2_des = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(img1_des, img2_des, k = k_val)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])

    result_img = cv2.drawMatchesKnn(img1, img1_kp, img2, img2_kp, good_matches, None, flags=0)

    similarity = len(good_matches) / len(matches)

    return result_img, similarity

def image_process(images, ratio, k_val):
    img1 = images[0]
    img2 = images[1]
    image, results = calculate_similarity(img1, img2, ratio, k_val)
    return image, results

class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Template Matching'
    node_tag = 'Template_Matching'

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
        tag_node_output03_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output03'
        tag_node_output03_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output03Value'
        tag_node_output04_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output04'
        tag_node_output04_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output04Value'
        tag_node_output05_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output05'
        tag_node_output05_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output05Value'
        tag_node_output06_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output06'
        tag_node_output06_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output06Value'
        tag_node_output07_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output07'
        tag_node_output07_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output07Value'

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
            # 画像入力端子1
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Template image',
                )
            # 画像入力端子2
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input02_value_name,
                    default_value='Target image',
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
            # 類似度
            with dpg.node_attribute(
                    tag=tag_node_output03_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=tag_node_output03_value_name,
                    default_value='similarity',
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
        output_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output03Value'
        output_value04_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output04Value'
        output_value05_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output05Value'
        output_value06_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output06Value'
        output_value07_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output07Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        value = 0

        # 接続情報確認
        connection_info_src = ['', '']
        count = 0
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_IMAGE:
                # 画像取得元のノード名(ID付き)を取得
                connection = connection_info[0]
                connection = connection.split(':')[:2]
                connection_info_src[count] = ':'.join(connection)
                count += 1

        # パラメータ
        ratio = float(dpg_get_value(input_value03_tag))
        k = int(dpg_get_value(input_value04_tag))
        thres = float(dpg_get_value(input_value05_tag))

        # 画像取得
        frames = [0, 0]
        for index in range(len(connection_info_src)):
            frames[index] = node_image_dict.get(connection_info_src[index], None)

        image = frames[0]

        # 計測開始
        if frames[0] is not None and frames[1] is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frames[0] is not None and frames[1] is not None:
            image, results = image_process(frames, ratio, k)
            results = round(results, 3)

        # 計測終了
        if frames[0] is not None and frames[1] is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')
            dpg_set_value(output_value03_tag,
                          results)

        # 描画
        if frames[0] is not None and frames[1] is not None:
            texture = convert_cv_to_dpg(
                image,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(output_value01_tag, texture)

        return image, None

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

