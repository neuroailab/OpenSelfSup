from mmcv import Config
from openselfsup.models import build_model
from openselfsup.datasets import build_dataset
import os
from torch.utils.data import DataLoader
import pdb
from torch.utils.data import RandomSampler
import torch


MOCO_CFG_PATH = os.path.expanduser('~/openselfsup/configs/selfsup/moco/r18_v2.py')
LINEAR_CFG_PATH = os.path.expanduser('~/openselfsup/configs/benchmarks/linear_classification/imagenet/r18_moco_sl.py')
MOCO_MODEL_PATH = '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/selfsup/moco/r18_v2/epoch_200.pth'

cfg = Config.fromfile(LINEAR_CFG_PATH)
cfg.data.train.data_source.memcached = False
dataset = build_dataset(cfg.data.train)
data_loader = DataLoader(
        dataset,
        batch_size=50,
        sampler=RandomSampler(dataset),
        pin_memory=False)
model = build_model(Config.fromfile(MOCO_CFG_PATH).model)
model_dict = torch.load(MOCO_MODEL_PATH)
model.load_state_dict(model_dict['state_dict'])
model = model.encoder_q.cuda()
for param in model.parameters():
    param.requires_grad = False
model.eval()

all_labels = []
all_embds = []

for idx, data in enumerate(data_loader):
    all_labels.append(data['gt_label'])
    embd = model(data['img'].cuda())
    all_embds.append(embd[0].cpu().numpy())
    if idx % 100 == 0:
        print(idx)
    if idx >= 1000:
        break

SAVE_PATH = '/mnt/fs4/chengxuz/openselfsup_models/moco_r18_in_embds.pkl'
import pickle
pickle.dump(
        {'lbl': all_labels, 'embds': all_embds},
        open(SAVE_PATH, 'wb'))
