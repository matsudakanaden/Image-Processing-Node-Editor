#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg

from matplotlib import pyplot as plt

import io

def image_process(images):
    img1 = images[0] / 255
    img2 = images[1] / 255
    try:
        mssim, ssim = cv2.quality.QualitySSIM_compute(img1, img2)
        similarity = float(sum(mssim[:3]) / 3)

        # RGBの方向で平均を計算する
        ssim_image = 1 - ssim.mean(axis=2)
        # SSIMの結果を0-255の範囲に正規化
        ssim_normalized = (ssim_image * 255).astype(np.uint8)
        # カラーマップの適用
        ssim_colored = cv2.applyColorMap(ssim_normalized, cv2.COLORMAP_JET)

        return ssim_colored, similarity
    except:
        return None, 0

class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'SSIM Similarity'
    node_tag = 'SSIM Similarity'

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
            # 画像入力端子1
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input01_value_name,
                    default_value='Input BGR image 1',
                )
            # 画像入力端子2
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=tag_node_input02_value_name,
                    default_value='Input BGR image 2',
                )
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
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
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        output_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output03Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']

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

        # 画像取得
        frames = [0, 0]
        for index in range(len(connection_info_src)):
            frames[index] = node_image_dict.get(connection_info_src[index], None)

        image = frames[0]

        # 計測開始
        if frames[0] is not None and frames[1] is not None and use_pref_counter:
            start_time = time.perf_counter()

        if frames[0] is not None and frames[1] is not None:
            image, similarity = image_process(frames)

        # 計測終了
        if frames[0] is not None and frames[1] is not None and use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_value02_tag,
                          str(elapsed_time).zfill(4) + 'ms')
            dpg_set_value(output_value03_tag,
                          round(similarity, 3))

        # 描画
        if image is not None:
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

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
