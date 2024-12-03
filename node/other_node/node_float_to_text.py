#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg


def image_process(image):
    return image


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Float to Text'
    node_tag = 'FloatToText'

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
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input01Value'
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_TEXT + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Output01Value'

        # 設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = int(self._opencv_setting_dict['process_width'] / 2)
        small_window_h = int(self._opencv_setting_dict['process_height'] / 2)

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # 入力
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=tag_node_input01_value_name,
                    label="Input value",
                    width=small_window_w,
                    default_value=0,
                    callback=callback,
                )
            # 書式
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_text(
                    tag=tag_node_input02_value_name,
                    label="Format spec",
                    width=small_window_w,
                    default_value=".3f",
                    callback=callback,
                )
            # 出力
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=tag_node_output01_value_name,
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
        input_value01_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input01Value'
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Output01Value'

        # 接続情報確認
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_FLOAT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = float(dpg_get_value(source_tag))
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_TEXT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = dpg_get_value(source_tag)
                dpg_set_value(destination_tag, input_value)

        value = dpg_get_value(input_value01_tag)
        spec = dpg_get_value(input_value02_tag)
        
        try:
            text = format(value, spec)
        except Exception as e:
            text = str(value)

        dpg_set_value(input_value01_tag,
                        value)
        dpg_set_value(output_value01_tag,
                        text)

        return None, text

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input02_value = dpg_get_value(input_value02_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = tag_node_input02_value

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        tag_node_input02_value = setting_dict[input_value02_tag]

        dpg_set_value(input_value02_tag, tag_node_input02_value)
