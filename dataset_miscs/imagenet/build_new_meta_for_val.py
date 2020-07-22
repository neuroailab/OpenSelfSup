import os


old_meta_path = 'data/imagenet/meta/val_labeled.txt'
new_meta_path = 'data/imagenet/meta/val_labeled_new.txt'
val_folder = 'data/imagenet/val'

with open(old_meta_path, 'r') as fin:
    old_metas = fin.readlines()

old_metas = [_meta[:-1] for _meta in old_metas]
all_cates = os.listdir(val_folder)
all_cates.sort()
new_meta = []
for _meta in old_metas:
    jpg_path, lbl = _meta.split(' ')
    lbl = int(lbl)
    _cate = all_cates[lbl]
    new_jpg_path = os.path.join(_cate, jpg_path)
    if not os.path.exists(os.path.join(val_folder, new_jpg_path)):
        raise NotImplementedError
    new_meta.append(new_jpg_path + ' ' + str(lbl) + '\n')

with open(new_meta_path, 'w') as fout:
    fout.writelines(new_meta)
