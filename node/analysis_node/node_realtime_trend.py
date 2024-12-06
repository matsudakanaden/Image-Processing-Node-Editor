#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info

data_len: int = 100

def update_array(data: np.ndarray, value) -> np.ndarray:
    data[:data_len - 1] = data[1:data_len]
    data[data_len - 1] = value

    return data

class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Realtime Trend'
    node_tag = 'RealtimeTrend'

    elapsed_time: float = 0.0
    interval: float = 1.00

    x_data: np.ndarray = np.linspace(0, data_len - 1, data_len)
    y1_data = np.empty(data_len)
    y1_data[:] = np.nan
    y2_data = np.empty(data_len)
    y2_data[:] = np.nan
    y3_data = np.empty(data_len)
    y3_data[:] = np.nan

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
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'

        tag_node_trend_name = tag_node_name + ':' + ':TrendValue'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            with dpg.node_attribute(
                    tag=tag_node_trend_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                # プロット
                with dpg.plot(
                    label='Trend',
                    width=640,
                    height=400,
                    show=True):
                    # 凡例
                    dpg.add_plot_legend(horizontal=True,
                                        location=dpg.mvPlot_Location_NorthEast)

                    # X軸(時間)
                    dpg.add_plot_axis(dpg.mvXAxis, label='x', tag='x_axis', time=True)

                    # Y軸1
                    dpg.add_plot_axis(dpg.mvYAxis, label='y1', tag='y1')
                    dpg.add_line_series(self.x_data, self.y1_data, label='y1', parent=dpg.last_item(), tag='y_axis1')

                    # Y軸2
                    dpg.add_plot_axis(dpg.mvYAxis, label='y2', tag='y2')
                    dpg.add_line_series(self.x_data, self.y2_data, label='y2', parent=dpg.last_item(), tag='y_axis2')

                    # Y軸3
                    dpg.add_plot_axis(dpg.mvYAxis, label='y3', tag='y3')
                    dpg.add_line_series(self.x_data, self.y3_data, label='y3', parent=dpg.last_item(), tag='y_axis3')
            # 入力1
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=tag_node_input01_value_name,
                    label="y1",
                    width=small_window_w - 30,
                    default_value=0,
                    callback=callback,
                )
            # 入力2
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=tag_node_input02_value_name,
                    label="y2",
                    width=small_window_w - 30,
                    default_value=0,
                    callback=callback,
                )
            # 入力3
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=tag_node_input03_value_name,
                    label="y3",
                    width=small_window_w - 30,
                    default_value=0,
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
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value01_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input01Value'
        input_value02_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']

        # 接続情報確認
        node_name = ''
        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_FLOAT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = float(dpg_get_value(source_tag))
                dpg_set_value(destination_tag, input_value)

        total_time = dpg.get_total_time()
        if total_time - self.elapsed_time >= self.interval:
            self.elapsed_time = total_time

            y1 = float(dpg_get_value(input_value01_tag))
            y2 = float(dpg_get_value(input_value02_tag))
            y3 = float(dpg_get_value(input_value03_tag))

            self.y1_data = update_array(self.y1_data, y1)
            self.y2_data = update_array(self.y2_data, y2)
            self.y3_data = update_array(self.y3_data, y3)

            dpg.set_value('y_axis1', [self.x_data, self.y1_data])
            dpg.set_value('y_axis2', [self.x_data, self.y2_data])
            dpg.set_value('y_axis3', [self.x_data, self.y3_data])

        #dpg.render_dearpygui_frame()

        return None, None

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value01_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input01Value'
        input_value02_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value01_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input01Value'
        input_value02_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input03Value'
