#JoyWaApi.dll用Pythonライブラリ
#32ビット版Pythonのみ対応(JoyWaApi.dllが32ビット版のため)

from ctypes import *
import ctypes
import struct
import sys
from typing import TypeVar, Union

class _U(ctypes.Union):
    _fields_ = [("dblVal", c_double),
                ("pbVal", c_char * 16)]

class TCOM_DATA1(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("col_id", c_long),
                ("u", _U),
                ("dtype", c_char)]

tagdict = dict()

#DLL読込
api = cdll.LoadLibrary(r"C:\Windows\SysWOW64\JoyWaApi.dll")

MAX_TAG: c_long = 1
NAME_LEN: c_long = 256

#JWサーバ接続
def Connect():
    return api.ConnectNet()

#JWサーバ切断
def Disconnect():
    return api.DisconnectNet()

#タグ値読込
def Read(tagname):
    tagid = GetTagId(tagname)
    
    pCom = TCOM_DATA1()
    pCom.col_id = tagid
    pCom.dblVal = 0
    pCom.dtype = 0

    api.JWRead(
        0,
        "",
        1,
        ctypes.pointer(pCom))

    if pCom.dtype != b'\x05':
        return pCom.dblVal
    else:
        return pCom.pbVal.decode("shift_jis")

#タグ値書込(数値)
def WriteValue(tagname, value):
    tagid = GetTagId(tagname)

    ids = (c_long * 1)(tagid)
    ids_tmp  = (c_long * len(ids))(*ids)
    ids_data = cast(ctypes.pointer(ids_tmp), ctypes.POINTER(c_long))

    writeval1 = struct.pack("d", value)
    writeval2 = (c_char * len(writeval1)).from_buffer_copy(writeval1)
    writeval_tmp  = (c_char * len(writeval2))(*writeval2)
    writeval_data = ctypes.cast(ctypes.pointer(writeval_tmp), ctypes.POINTER(c_char))
    
    ret = api.JWWrite(
        0,
        "",
        0,
        1,
        len(writeval1),
        ids_data,
        writeval_data)

    return ret

#タグ値書込(文字列・最大16バイト)
def WriteText(tagname, value):
    tagid = GetTagId(tagname)

    ids = (c_long * 1)(tagid)
    ids_tmp  = (c_long * len(ids))(*ids)
    ids_data = cast(ctypes.pointer(ids_tmp), ctypes.POINTER(c_long))

    value1 = struct.pack('{}s'.format(len(value.encode("shift_jis"))), value.encode("shift_jis"))
    value2 = (c_char * len(value1)).from_buffer_copy(value1)
    value_tmp  = (c_char * len(value2))(*value2)
    value_data = ctypes.cast(ctypes.pointer(value_tmp), ctypes.POINTER(c_char))
    
    ret = api.JWWrite(
        0,
        "",
        0,
        1,
        len(value1),
        ids_data,
        value_data)

    return ret

#ディクショナリからタグIDを取得
def GetTagId(tagname):
    if (tagname in tagdict):
        return tagdict[tagname]
    else:
        tagid = JWGetTagIDS2(tagname)
        if tagid == c_long(-1):
            raise NameError("タグ名が見つかりません")
        tagdict[tagname] = tagid
        return tagid

#JWからタグIDを取得
def JWGetTagIDS2(tagname):
    tagname_tmp = create_string_buffer(tagname.encode("ASCII"))
    tagname_data = cast(tagname_tmp, c_char_p)
    tagid= c_long()

    api.JWGetTagIDS2(
        1,
        tagname_data,
        NAME_LEN,
        0,
        byref(tagid),
        sys.getsizeof(c_long),
        0,
        -1,
        -1)

    return tagid
