meta_file = '/mnt/fs4/chengxuz/Dataset/yfcc/meta.txt'
meta_file_dst = '/mnt/fs4/chengxuz/Dataset/yfcc/meta_short.txt'

with open(meta_file, 'r') as fin:
    all_jpgs = fin.readlines()
all_jpgs = [_jpg[len('/mnt/fs4/Dataset/YFCC/images/'):] for _jpg in all_jpgs]
with open(meta_file_dst, 'w') as fout:
    fout.writelines(all_jpgs)
