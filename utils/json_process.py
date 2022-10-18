# -- coding: utf-8 --

import json

# @Time : 2022/10/9 22:29
# @Author : Zhiheng Feng
# @File : json_process.py
# @Software : PyCharm



def json2dict(path: str) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def dict2json(data: dict, path: str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    path = r'F:\MyWork\Lab\Code\PSFNet\data\vel_to_cam_Tr.json'
    dd = json2dict(path)
    print(dd)







