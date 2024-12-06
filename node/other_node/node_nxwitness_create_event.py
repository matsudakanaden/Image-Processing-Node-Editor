#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
from node_editor.util import convert_cv_to_dpg
from node.draw_node.draw_util.draw_util import draw_info

import requests
import json

#ログイン
def login(server_url, username, password):
    r = requests.post(server_url + '/rest/v2/login/sessions',
        headers = {
            'accept': 'application/json'
        },
        json = {
            'username': username,
            'password': password,
            'setCookie': False
        },
        verify = False)
    result = r.json()
    return result

#汎用イベント登録
def create_event(server_url, token, source, caption, description):
    r = requests.post(server_url + '/api/createEvent/',
        headers = {'content-type' : 'application/json', 'Authorization': 'Bearer {}'.format(token)},
        json = {
            'source': source,
            'caption': caption,
            'description': description
        },
        verify = False)
    
    return r

#ログアウト
def logout(server_url, token):
    r = requests.delete(server_url + '/rest/v2/login/sessions/{}'.format(token),
        headers = {'content-type' : 'application/json', 'Authorization': 'Bearer {}'.format(token)},
        verify = False)

    return r

def create_event(server_url, username, password, source, caption, description):
    try:
        login_info = login(server_url, username, password)
        create_event(server_url, login_info['token'], source, caption, description)
        logout(server_url, login_info['token'])
    except Exception as e:
        print(e)

class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'Create Event'
    node_tag = 'NxwCreateEvent'

    _opencv_setting_dict = None

    source = 'source_test'
    caption = 'caption_test'
    description = 'description_test'

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
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input03Value'
        tag_node_input04_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input04'
        tag_node_input04_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input04Value'
        tag_node_input05_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input05'
        tag_node_input05_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input05Value'
        tag_node_input06_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input06'
        tag_node_input06_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input06Value'

        tag_node_button_name = tag_node_name + ':' + self.TYPE_TEXT + ':Button'
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']

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
                format=dpg.mvFormat_Float_rgb,
            )

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # サーバーURL
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_text(
                    tag=tag_node_input01_value_name,
                    label="Server URL",
                    width=small_window_w - 30,
                    default_value='https://127.0.0.1:7001/',
                )
            # ユーザー名
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_text(
                    tag=tag_node_input02_value_name,
                    label="User name",
                    width=small_window_w - 30,
                    default_value='admin',
                )
            # パスワード
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_text(
                    tag=tag_node_input03_value_name,
                    label="Password",
                    width=small_window_w - 30,
                    default_value='NxAdmin',
                    password=True,
                )
            # ソース
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_text(
                    tag=tag_node_input04_value_name,
                    label="Source",
                    width=small_window_w - 30,
                    default_value=self.source,
                    #callback=self.button_callback,
                    #user_data=tag_node_name,
                )
            # キャプション
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_text(
                    tag=tag_node_input05_value_name,
                    label="Caption",
                    width=small_window_w - 30,
                    default_value=self.caption,
                    #callback=self.button_callback,
                    #user_data=tag_node_name,
                )
            # デスクリプション
            with dpg.node_attribute(
                    tag=tag_node_input06_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_text(
                    tag=tag_node_input06_value_name,
                    label="Description",
                    width=small_window_w - 30,
                    default_value=self.description,
                    #callback=self.button_callback,
                    #user_data=tag_node_name,
                )
            # 実行ボタン
            with dpg.node_attribute(
                    tag=tag_node_button_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label="Create Event",
                    tag=tag_node_button_value_name,
                    width=small_window_w - 30,
                    callback=self.button_callback,
                    user_data=tag_node_name,
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

        # 接続情報確認
        node_name = ''
        connection_info_src = ''
        for connection_info in connection_list:
            connection_type = connection_info[0].split(':')[2]
            if connection_type == self.TYPE_TEXT:
                # 接続タグ取得
                source_tag = connection_info[0] + 'Value'
                destination_tag = connection_info[1] + 'Value'
                # 値更新
                input_value = dpg_get_value(source_tag)
                dpg_set_value(destination_tag, input_value)

        # 値変化チェック
        input_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input05Value'
        input_value06_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input06Value'
        server_url = dpg_get_value(input_value01_tag)
        username = dpg_get_value(input_value02_tag)
        password = dpg_get_value(input_value03_tag)
        new_source = dpg_get_value(input_value04_tag)
        new_caption = dpg_get_value(input_value05_tag)
        new_description = dpg_get_value(input_value06_tag)
        if new_source != self.source or new_caption != self.caption or new_description != self.description:
            create_event(server_url, username, password, new_source, new_caption, new_description)
            self.source = new_source
            self.caption = new_caption
            self.description = new_description

        return None, None

    def close(self, node_id):
        pass

    def button_callback(self, sender, data, user_data):
        tag_node_name = user_data
        input_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input05Value'
        input_value06_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input06Value'

        server_url = dpg_get_value(input_value01_tag)
        username = dpg_get_value(input_value02_tag)
        password = dpg_get_value(input_value03_tag)
        source = dpg_get_value(input_value04_tag)
        caption = dpg_get_value(input_value05_tag)
        description = dpg_get_value(input_value06_tag)

        create_event(server_url, username, password, source, caption, description)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        input_value01 = dpg_get_value(input_value01_tag)
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value02 = dpg_get_value(input_value02_tag)
        input_value03_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input03Value'
        input_value03 = dpg_get_value(input_value03_tag)
        input_value04_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input04Value'
        input_value04 = dpg_get_value(input_value04_tag)
        input_value05_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input05Value'
        input_value05 = dpg_get_value(input_value05_tag)
        input_value06_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input06Value'
        input_value06 = dpg_get_value(input_value06_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value01_tag] = input_value01
        setting_dict[input_value02_tag] = input_value02
        setting_dict[input_value03_tag] = input_value03
        setting_dict[input_value04_tag] = input_value04
        setting_dict[input_value05_tag] = input_value05
        setting_dict[input_value06_tag] = input_value06

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value01_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input05Value'
        input_value06_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input06Value'

        input_value01 = setting_dict[input_value01_tag]
        input_value02 = setting_dict[input_value02_tag]
        input_value03 = setting_dict[input_value03_tag]
        input_value04 = setting_dict[input_value04_tag]
        input_value05 = setting_dict[input_value05_tag]
        input_value06 = setting_dict[input_value06_tag]

        dpg_set_value(input_value01_tag, input_value01)
        dpg_set_value(input_value02_tag, input_value02)
        dpg_set_value(input_value03_tag, input_value03)
        dpg_set_value(input_value04_tag, input_value04)
        dpg_set_value(input_value05_tag, input_value05)
        dpg_set_value(input_value06_tag, input_value06)
