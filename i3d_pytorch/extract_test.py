import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import numpy as np
import pickle as pkl
import torch
from torch.nn.modules import Module
import torch.utils.data
import torchvision
from torchvision import datasets, transforms

from pytorch_i3d import InceptionI3d
import videotransforms
from charades_dataset_full import Charades_for_TC as Dataset

load_model = './models/rgb_charades.pt'
mode = 'rgb'
root='./Charades/Charades_v1_rgb'
split = './Charades/charades.json'
batch_size = 1
save_dir = './features_i3d'
# setup dataset
print('dataset prepared.')
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

dataset = Dataset(split, 'training', root, mode, save_dir, test_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                         pin_memory=False)

val_dataset = Dataset(split, 'testing', root, mode, save_dir, test_transforms)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                             pin_memory=False)

dataloaders = {'train': dataloader, 'val': val_dataloader}
datasets = {'train': dataset, 'val': val_dataset}
# setup the model
print("loading model...")
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(157)
i3d.load_state_dict(torch.load(load_model))

for phase in ['train', 'val']:
    i3d.eval()  # Set model to evaluate mode

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    # Iterate over data.
    for data in dataloaders[phase]:
        # get the inputs
        inputs, labels, name = data
        if os.path.exists(os.path.join(save_dir, name[0] + '.pkl')):
            print("feature file %s exist." % (name[0]+'.pkl'))
            continue

        b, c, t, h, w = inputs.shape
        if t > 1600:
            features = []
            for start in range(1, t - 56, 1600):
                end = min(t - 1, start + 1600 + 56)
                start = max(1, start - 48)
                ip = torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda()
                features.append(i3d.mixed_5c_output(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
            with open(os.path.join(save_dir, name[0]+'.pkl'), 'wb') as f:
                pkl.dump(np.concatenate(features, axis=0), f)
            print("write to %s done." % (name[0]+'.pkl'))
            # np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
        else:
            # wrap them in Variable
            # inputs = inputs.cuda()
            features = i3d.mixed_5c_output(inputs).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
            with open(os.path.join(save_dir, name[0]+'.pkl'), 'wb') as f:
                pkl.dump(features, f)
            print("write to %s done." % (name[0] + '.pkl'))

            # np.save(os.path.join(save_dir, name[0]), features)

