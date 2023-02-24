import cv2 as cv
import os
import numpy as np
from utils.general import get_data_root
import argparse
import time
from datasets.testdataset import configdataset
from datasets.genericdataset import ImagesFromList
import torch
import torchvision.transforms as transforms


test_datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Feature Attention Training')

parser.add_argument('directory', metavar='EXPORT_DIR',
                    help='destination where trained descriptors should be saved')
parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                    help='comma separated list of test datasets: ' +
                         ' | '.join(test_datasets_names) +
                         ' (default: roxford5k,rparis6k)')


def main():
    global args
    args = parser.parse_args()
    
    # manually check if there are unknown test datasets
    for dataset in args.test_datasets.split(','):
        if dataset not in test_datasets_names:
            raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))
        
    args.directory = os.path.join(args.directory, 'sift')
        
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        
    datasets = args.test_datasets.split(',')
    output_path = args.directory
    image_size = 1024
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
        
    for dataset in datasets:
        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

        # extract database and query vectors
        output_path = os.path.join(args.directory, dataset, 'data')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('>> {}: database images...'.format(dataset))
        extract_sift(output_path, images, image_size, transform=transform)  # implemented with torch.no_grad
        
        output_path = os.path.join(args.directory, dataset, 'query')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('>> {}: query images...'.format(dataset))
        extract_sift(output_path, qimages, image_size, bbxs, transform=transform)  # implemented with torch.no_grad
        
        
def extract_sift(output_path, images, image_size, bbxs=None, transform=None, print_freq=10):
    
    output_path_des = os.path.join(output_path, 'des')
    output_path_kp = os.path.join(output_path, 'kp')
    if not os.path.exists(output_path_des):
        os.makedirs(output_path_des)
    if not os.path.exists(output_path_kp):
        os.makedirs(output_path_kp)
        
    # print('output_path: {}, {}'.format(output_path_des, output_path_kp))

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():

        for i, input in enumerate(loader):
            # print('output_path_des/kp: {}, {}'.format(output_path_des, output_path_kp))
            des_path = os.path.join(output_path_des, images[i].split('/')[-1] + '.des')
            kp_path = os.path.join(output_path_kp, images[i].split('/')[-1] + '.kp')
            input = input.cuda()
            # print('des_path, kp_path: {}, {}'.format(des_path, kp_path))
            extract_sift_features(des_path, kp_path, input)

            if (i + 1) % print_freq == 0 or (i + 1) == len(images):
                print('\r>>>> {}/{} done...'.format((i + 1), len(images)), end='')
        print('')


def extract_sift_features(des_path, kp_path, input):
    input = input.cpu().squeeze().permute((1, 2, 0))[:, :, [2, 1, 0]].numpy()
    input = input * 255
    input = input.astype(np.uint8)
    sift = cv.xfeatures2d.SIFT_create(nfeatures=500, contrastThreshold=0.01, edgeThreshold=30, sigma=1.6)
    kps, des = sift.detectAndCompute(input, None)
    des_np = des.astype(np.uint8)
    kps = np.array([x.pt for x in kps]).astype(np.float16)
    
    # print('saving to {}, {}...'.format(des_path, kp_path))
    torch.save(des_np, des_path)
    torch.save(kps, kp_path)
    

if __name__ == '__main__':
    main()


