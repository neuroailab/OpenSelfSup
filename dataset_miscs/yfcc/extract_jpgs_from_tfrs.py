import sys, os
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm
import pdb


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate tfrecords from jpeg images')
    parser.add_argument(
            '--tfr_dir', 
            default='/mnt/fs4/chengxuz/Dataset/yfcc/tfrs', type=str, 
            action='store', help='Directory to save the tfrecords')
    parser.add_argument(
            '--meta_file', 
            default='/mnt/fs4/chengxuz/Dataset/yfcc/meta_short.txt', 
            type=str, action='store', help='Meta file listing the jpgs')
    parser.add_argument(
            '--random_seed', 
            default=0, type=int, 
            action='store', help='Random seed for numpy')
    parser.add_argument(
            '--target_dir', 
            default='/data5/chengxuz/Dataset/yfcc/imagenet_size_jpgs_from_tfr', type=str, 
            action='store', help='Directory to save the tfrecords')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.system('mkdir -p %s' % args.target_dir)

    with open(args.meta_file, 'r') as fin:
        all_jpgs = fin.readlines()
    all_jpgs = [_jpg[:-1] for _jpg in all_jpgs]
    target_jpgs = [
            os.path.join(args.target_dir, _jpg) \
            for _jpg in all_jpgs]

    np.random.seed(args.random_seed)
    target_jpgs = np.random.permutation(target_jpgs)

    all_dirs = np.unique([os.path.dirname(_jpg) for _jpg in target_jpgs])
    for _dir in tqdm(all_dirs):
        os.system('mkdir -p ' + _dir)

    tfr_paths = os.listdir(args.tfr_dir)
    tfr_paths.sort()
    tfr_paths = [
            os.path.join(args.tfr_dir, _path) \
            for _path in tfr_paths]

    idx_now = 0
    for input_file in tqdm(tfr_paths):
        input_iter = tf.python_io.tf_record_iterator(path=input_file)

        for curr_record in input_iter:
            image_example = tf.train.Example()
            image_example.ParseFromString(curr_record)

            img_string = image_example.features.feature['images'].bytes_list.value[0]
            with open(target_jpgs[idx_now], 'wb') as fout:
                fout.write(img_string)
            idx_now += 1
        input_iter.close()


if __name__=="__main__":
    main()
