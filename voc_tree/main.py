import voc_tree.Dataset as Dataset
from voc_tree.VocTree import VocTree
from utils.evaluate import compute_map_and_print
from datasets.testdataset import configdataset
import numpy as np
import pickle as pkl
import os
import argparse
from utils.general import get_data_root
import time

test_datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']

parser = argparse.ArgumentParser(description='voc tree')

parser.add_argument('directory', metavar='EXPORT_DIR',
                    help='destination where trained networks should be saved')
parser.add_argument('--root-directory', metavar='SOURCE_DIR',
                    help='root of the descriptor files', default='./features/sift')
parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                    help='comma separated list of test datasets: ' +
                         ' | '.join(test_datasets_names) +
                         ' (default: roxford5k,rparis6k)')
parser.add_argument('--tree-height', '-th', default=6, type=int, metavar='N',
                    help='voc tree max height')
parser.add_argument('--tree-branch', '-tb', default=10, type=int, metavar='N',
                    help='voc tree branch number')
parser.add_argument('--no-homo', dest='val', action='store_false',
                    help='whether is findHomography')
parser.add_argument('--no-ransac', dest='val', action='store_false',
                    help='no ransac reranking')
parser.add_argument('--no-train', dest='val', action='store_false',
                    help='train the tree')
parser.add_argument('--output', metavar='OUTPUT_FN',
                    help='name of the output file for test results', default='output_sift.txt')
parser.add_argument('--rerank-num', '-rn', default=300, type=int, metavar='N',
                    help='number of images reranked')

def main():
    global args, f
    args = parser.parse_args()
    f = open(args.output, 'w')
    
    tree_path = args.directory
    branchs = args.tree_branch
    height = args.tree_height
    
    if not os.path.exists(tree_path):
        os.makedirs(tree_path)
    
    is_H = True
    Ransac = True
    n_rerank = args.rerank_num

    for dataset in args.test_datasets.split(','):
        if dataset not in test_datasets_names:
            raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))
        
    datasets = args.test_datasets.split(',')
    for dataset in datasets:
        root_path = os.path.join(args.root_directory, dataset)
        des_dataset = Dataset.DescriptorDataset(root_path=root_path)
        
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))

        start = time.time()
        vocTree = VocTree(Dataset=des_dataset,
                                Tree_path=os.path.join(tree_path, dataset + '_tree_{}_{}'.format(branchs, height)),
                                BoFs_path=os.path.join(tree_path, dataset + '_BoFs_{}_{}'.format(branchs, height)),
                                Train=False,
                                branchs=branchs,
                                maximum_height=height,
                                gnd=cfg)

        f.write('>> Training finished in {} s\n'.format(time.time() - start))
        ranks = []
        gnt = cfg['gnd']
        for i in range(len(gnt)):
            print('\r>>{}/70'.format(i + 1), end='')
            Q_ID = i
            
            rank = vocTree.Query(Q_image_ID=Q_ID,
                                root_node=vocTree.Tree,
                                BoFs=vocTree.BoFs,
                                result_size=10000)

            if Ransac:
                rerank_list = vocTree.reRank(rank[:n_rerank], Q_ID, is_H)
                rank[:n_rerank] = rerank_list

            rank = rank.reshape(-1, 1)
            ranks.append(rank)

        print('')

        ranks = np.concatenate(ranks, axis=1)

        compute_map_and_print(dataset, ranks, gnt, f=f)


if __name__ == '__main__':
    main()