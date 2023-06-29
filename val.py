import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import main

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from main import MDUNet

import pdb


import numpy as np
import matplotlib.pyplot as plt
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])

    
    model=main.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    
    
    
    model = model.cuda()

    
    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=3407)
    

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']),strict=False)
    
    # model.encoder1.register_forward_hook(get_activation('encoder1'))
    

    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        
      
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)#val_transform
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            #print(len(val_dataset))
            #pdb.set_trace()
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            # output = model(input)
            outputs = model(input)
            output=outputs[-1]
            # output=outputs


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
            print(len(output))

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))
            
            
            act = activation['encoder1'].squeeze()
            print (act.size())
            print (act.max())
            print (act.min())
            act = (act-act.min())/(act.max()-act.min())
            print (act.max())
            print (act.min())
            #raise
            row_n = 8
            for img_i in range(act.size(0)):
                fig, axarr = plt.subplots(row_n, act.size(1)//row_n,figsize = [8,8])#,gridspec_kw = {'wspace':0.02, 'hspace':0.02}
                # plt.tight_layout() #使子图紧凑排放
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.023, hspace=0.02)
                for idx_x in range(row_n):
                    for idx_y in range(act.size(1)//row_n):
                        axarr[idx_x, idx_y].imshow(act[img_i, idx_x*row_n+idx_y].cpu(),aspect='auto')
                        axarr[idx_x, idx_y].axis('off')
                        
                #plt.show()
                save_path = os.path.join('outputs', config['name'], 'feature_img_train_encoder1')
                os.makedirs(save_path, exist_ok=True)
                f_name = meta['img_id'][img_i]
                plt.savefig(os.path.join(save_path, f'{f_name}.jpg'))
    
    
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
