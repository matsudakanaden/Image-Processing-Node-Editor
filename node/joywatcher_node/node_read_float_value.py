#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC

#JoyWaApi.dll用Pythonライブラリのインポート
#32ビット版Pythonのみ対応(JoyWaApi.dllが32ビット版のため)
import JoyWatcherAccessor as jw

class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Read JoyWatcher Tag Value'
    node_tag = 'JWReadValue'

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
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Output01Value'

        # 設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['input_window_width']
        jw_tagname = self._opencv_setting_dict['jw_tagname']

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # JoyWatcherタグ名入力欄
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_text(
                    tag=tag_node_output01_value_name,
                    label="Tagname",
                    width=small_window_w - 94,
                    default_value=tagname,
                    callback=callback,
                )
                  
        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        jw.Connect()
        value = jw.Read(jw_tagname)
        return None, value

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        output_value_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output01Value'

        output_value = round((dpg_get_value(output_value_tag)), 3)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[output_value_tag] = output_value

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        output_value_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Output01Value'

        output_value = float(setting_dict[output_value_tag])

        dpg_set_value(output_value_tag, output_value)
