"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from termcolor import colored
from mireczech.models import SimCLRModel

@torch.no_grad()
def mireczech_fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model.backbone(images).flatten(start_dim=1)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    val_transforms = get_val_transformations(p)
    val_dataset = get_val_dataset(p, val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)
    # print('Train transforms:', train_transforms)
    print('Validation transforms:', val_transforms)
    # print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], None)
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], None)
    memory_bank_val.cuda()

    # Load final model
    state_dict = torch.load(p['pretext_model'], map_location='cpu')['state_dict']
    for k in list(state_dict.keys()):
        if not k.startswith('backbone.'):
            del state_dict[k]

    model = SimCLRModel()
    model.load_state_dict(state_dict)
    model.cuda()

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    mireczech_fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)   

   
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    mireczech_fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)   

 
if __name__ == '__main__':
    main()
