import os
import numpy as np
import torch


class DescriptorDataset(object):
    def __init__(self, root_path=None):
        super(DescriptorDataset, self).__init__()
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'data')
        self.query_path = os.path.join(root_path, 'query')
        self.data_des_path = os.path.join(self.data_path, 'des')
        self.data_kp_path = os.path.join(self.data_path, 'kp')
        self.query_des_path = os.path.join(self.query_path, 'des')
        self.query_kp_path = os.path.join(self.query_path, 'kp')

        self.N_images = len(os.listdir(self.data_des_path))


    def DB_features(self, is_query, imlist):
        index = []  # number of des in images
        Des_to_Im = []
        Descriptors = []
        kps = []
        idxs = []  # start indx of image des
        
        if is_query is True:
            des_path = self.query_des_path
            kp_path = self.query_kp_path
        else:
            des_path = self.data_des_path
            kp_path = self.data_kp_path

        curr = 0

        for k, image_fn in enumerate(imlist):
            des = torch.load(os.path.join(des_path, image_fn + '.jpg.des'))
            print(des.shape)
            Descriptors.append(des)
            index.append(len(des))
            idxs.append(curr)
            curr += len(des)
            Des_to_Im.extend([k]*len(des))
        idxs.append(curr)
        Descriptors = np.concatenate(Descriptors, axis=0)
        index = np.array(index)
        idxs = np.array(idxs)
        Des_to_Im = np.array(Des_to_Im)

        for image_fn in imlist:
            kp = torch.load(os.path.join(kp_path, image_fn + '.jpg.kp'))
            kps.append(kp)
        kps = np.concatenate(kps, axis=0)

        Descriptor_IDs = np.arange(Descriptors.shape[0])  # 就是从0开始的标号

        return index, Descriptors, kps, Descriptor_IDs, Des_to_Im, idxs
