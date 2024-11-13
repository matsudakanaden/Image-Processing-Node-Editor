#!/usr/bin/env python
# coding: utf-8

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

#ログイン情報取得
def get_login_session(server_url, token):
    r = requests.get(server_url + f'/rest/v2/login/sessions/{token}',
        headers = {
            'content-type' : 'application/json',
            'Authorization': 'Bearer {}'.format(token)
        },
        verify = False)
    result = r.json()
    return result

#サーバ情報取得
def read_server_info(server_url, token, server_id):
    r = requests.get(server_url + f'/rest/v2/servers/{server_id}/info',
        headers = {
            'content-type' : 'application/json'
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

#ブックマーク登録
def create_bookmark(server_url, token, device_id, name, description, start_datetime, end_datetime, tags):
    startEpoch = int(start_datetime * 1000)
    endEpoch = int(end_datetime * 1000)
    
    body = {}
    body['name'] = name
    body['startTimeMs'] = str(startEpoch)
    body['durationMs'] = str(endEpoch - startEpoch)
    if 0 < len(description):
        body['description'] = description
    if 0 < len(tags):
        body['tags'] = []
        for tag in tags:
            body['tags'].append(str(tag))

    r = requests.post(server_url + '/rest/v2/devices/{}/bookmarks/'.format(device_id),
        headers = {'content-type' : 'application/json', 'Authorization': 'Bearer {}'.format(token)},
        json = body,
        verify = False)

    return r

#ブックマーク登録(エポックミリ秒指定)
def create_bookmark_epoch(server_url, token, device_id, name, description, startEpoch, endEpoch, tags):
    body = {}
    body['name'] = name
    body['startTimeMs'] = str(startEpoch)
    body['durationMs'] = str(endEpoch - startEpoch)
    if 0 < len(description):
        body['description'] = description
    if 0 < len(tags):
        body['tags'] = []
        for tag in tags:
            body['tags'].append(str(tag))

    r = requests.post(server_url + '/rest/v2/devices/{}/bookmarks/'.format(device_id),
        headers = {'content-type' : 'application/json', 'Authorization': 'Bearer {}'.format(token)},
        json = body,
        verify = False)

    return r

#ログアウト
def logout(server_url, token):
    r = requests.delete(server_url + '/rest/v2/login/sessions/{}'.format(token),
        headers = {'content-type' : 'application/json', 'Authorization': 'Bearer {}'.format(token)},
        verify = False)

    return r
