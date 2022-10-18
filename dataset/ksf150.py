# -*- coding:UTF-8 -*-

import os
import glob
import os.path
import numpy as np
import torch.utils.data as data

from utils.json_process import json2dict
from dataset.sf_preprocess import RangeLimitWithSF, TransformerWithSF, RandomSampleWithSF

# author:Zhiheng Feng
# datetime:2022/2/25 19:08
# software: PyCharm

"""
文件说明：
    
"""
testIndex = [1, 5, 7, 8, 10, 12, 15, 17, 20, 21, 24, 25, 29, 30, 31, 32, 34, 35, 36, 39, 40, 44, 45, 47, 48, 49, 50, 51,
             53, 55, 56, 58, 59, 60, 70, 71, 72, 74, 76, 77, 78, 79, 81, 82, 88, 91, 93, 94, 95, 98]
trainIndex = [0, 2, 3, 4, 6, 9, 11, 13, 14, 16, 18, 19, 22, 23, 26, 27, 28, 33, 37, 38, 41, 42, 43, 46, 52, 54, 57, 61,
              62, 63, 64, 65, 66, 67, 68, 69, 73, 75, 80, 83, 84, 85, 86, 87, 89, 90, 92, 96, 97, 99, 100, 101, 102,
              103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
              124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
              145, 146, 147, 148, 149]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class KSF150(data.Dataset):
    # def __init__(self, root='../data', npoints=16384, train=True):
    def __init__(self, dataset_path, just_eval, train, pre_processes):
        super().__init__()
        self.train = train
        self.just_eval = just_eval
        self.root = dataset_path
        self.datapath = self.make_dataset()

        # 获取雷达坐标系到相机坐标系变换的矩阵
        self.vel2cam_dict = json2dict('data/ksf150_velo_to_cam.json')

        # 构建点云数据预处理
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def make_dataset(self):
        if self.just_eval:
            datapath = [os.path.join(self.root, '{:06d}.npz'.format(index)) for index in range(150)]
        else:
            if self.train == True:
                datapath = [os.path.join(self.root, '{:06d}.npz'.format(index)) for index in trainIndex]
            else:
                datapath = [os.path.join(self.root, '{:06d}.npz'.format(index)) for index in testIndex]
        return datapath


    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pos1 = data['pos1']
            pos2 = data['pos2']
            flow = data['gt']

        # 转换到相机坐标系下
        Tr = np.array(self.vel2cam_dict[f'Tr_{index:03d}'])
        pos1 = self.vel2cam(pos1, Tr)
        pos2 = self.vel2cam(pos2, Tr)
        flow = self.vel2cam(flow, Tr)

        data_dict = {'pc1':pos1, 'pc2':pos2, 'flow':flow}
        data_dict = self.apply_pre_processes(data_dict)
        pc1 =  data_dict['pc1'].astype(np.float32)
        pc2 =  data_dict['pc2'].astype(np.float32)
        flow =  data_dict['flow'].astype(np.float32)
        color1 = pc1
        color2 = pc2

        return {'pc1': pc1, 'pc2': pc2, 'norm1': color1, 'norm2': color2, 'flow': flow}

    def __len__(self):
        return len(self.datapath)


    def vel2cam(self, lidar, Tr):
        lidar =  np.concatenate([lidar, np.ones((lidar.shape[0], 1))], axis=-1)
        lidar = lidar @ Tr.T
        lidar = lidar[:,:3]
        return lidar


if __name__ == '__main__':
    import mayavi.mlab as mlab

    d = KSF150(dataset_path='kitti_rm_ground')
    print(len(d))
    import time

    tic = time.time()
    for i in range(1, 100):
        pc1, pc2, color1, color2, flow, mask = d[i]
        print(pc1.shape, pc2.shape)
        continue

        '''mlab.figure(bgcolor=(1,1,1))
        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        input()

        mlab.figure(bgcolor=(1,1,1))
        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], flow[:,0], flow[:,1], flow[:,2], scale_factor=1, color=(0,0,1), line_width=0.2)
        input()'''

    print(time.time() - tic)
    print(pc1.shape, type(pc1))
