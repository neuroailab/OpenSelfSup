import sys, os
import numpy as np
import argparse
from tqdm import tqdm
import pdb
import shutil


def get_parser():
    parser = argparse.ArgumentParser(
            description='Copy wantted jpgs from all images, ' \
                    + 'default parameters are for node7')
    parser.add_argument(
            '--target_dir', 
            default='/data5/chengxuz/Dataset/infant_headcam/imagenet_size_jpgs', 
            type=str, action='store', help='Directory to put the images')
    parser.add_argument(
            '--source_dir', 
            default='/data4/shetw/infant_headcam/jpgs_extracted', type=str, 
            action='store', help='Directory storing the original images')
    parser.add_argument(
            '--meta_file', 
            default='/mnt/fs4/chengxuz/Dataset/saycam_jpgs/SAYCam_jpgs.txt', 
            type=str, action='store', help='Meta file listing the jpgs')
    parser.add_argument(
            '--dataset_type', 
            default='saycam', 
            type=str, action='store', help='Dataset type')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.system('mkdir -p %s' % args.target_dir)

    with open(args.meta_file, 'r') as fin:
        all_jpgs = fin.readlines()
    all_jpgs = [_jpg[:-1] for _jpg in all_jpgs]
    if args.dataset_type == 'saycam':
        source_jpgs = []
        for _jpg in all_jpgs:
            if _jpg.startswith('A'):
                source_jpgs.append(
                        os.path.join(args.source_dir, 'Alicecam', _jpg))
            else:
                source_jpgs.append(
                        os.path.join(args.source_dir, 'Samcam', _jpg))
    elif args.dataset_type == 'yfcc':
        source_jpgs = [
                os.path.join(args.source_dir, _jpg) \
                for _jpg in all_jpgs]
    else:
        raise NotImplementedError()

    target_jpgs = [
            os.path.join(args.target_dir, _jpg) \
            for _jpg in all_jpgs]
    all_dirs = np.unique([os.path.dirname(_jpg) for _jpg in target_jpgs])
    for _dir in tqdm(all_dirs):
        os.system('mkdir -p ' + _dir)

    for src_jpg, dst_jpg in tqdm(zip(source_jpgs, target_jpgs)):
        shutil.copyfile(src=src_jpg, dst=dst_jpg)


if __name__=="__main__":
    main()
