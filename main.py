import os
import time
import argparse
import random
import math
from importlib import reload, import_module

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask, imagenet_a_mask

import torch    

import timm
import numpy as np

import tta_library.tent as tent
import tta_library.sar as sar
import tta_library.cotta as cotta
import tta_library.spa as spa
import tta_library.eata as eata
import tta_library.actmad as actmad
import tta_library.deyo as deyo

from tta_library.sam import SAM

from copy import deepcopy

from models.vpt import PromptViT


def validate_adapt(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    
    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            del output

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 39 == 0:
                logger.info(progress.display(i))
            
    return top1.avg, top5.avg

def obtain_train_loader(args):
    args.corruption = 'original'
    train_dataset, train_loader = prepare_test_data(args)
    train_dataset.switch_mode(True, False)
    return train_loader

# eata fisher
def get_fisher(args, net):
    net = deepcopy(net)
    args.corruption = 'original'
    fisher_dataset, fisher_loader = prepare_test_data(args)
    fisher_dataset.set_dataset_size(args.fisher_size)
    fisher_dataset.switch_mode(True, False)

    net = eata.configure_model(net)
    params, param_names = eata.collect_params(net)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)
        outputs = net(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in net.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    logger.info("compute fisher matrices finished")
    del ewc_optimizer

    return fishers, params


def use_spa_with_others(args, spa_net):
    valid_combinations = ['spa', 'spa+tent', 'spa+eta', 'spa+eata']
    assert args.algorithm in valid_combinations, NotImplementedError
    if 'tent' in args.algorithm:
        spa_net.use_tent = True
    if 'eata' in args.algorithm or 'eta' in args.algorithm:
        spa_net.use_eata = True
        fishers = None
        if 'eata' in args.algorithm:
            fishers, _ = get_fisher(args, spa_net.model)
        spa_net.fishers = fishers
        spa_net.fisher_alpha = args.fisher_alpha
        spa_net.current_model_probs = None
        spa_net.e_margin = args.e_margin
        spa_net.d_margin = args.d_margin

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_v2', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_sketch', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_adv', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--data_rendition', default='/dockerdata/imagenet-r', help='path to corruption dataset')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')
    parser.add_argument('--model', default='vitbase_timm', type=str, help='if shuffle the test set.')

    # algorithm selection
    parser.add_argument('--algorithm', default='foa', type=str, help='supporting foa, sar, cotta and etc.')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # model settings
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')

    # foa settings
    parser.add_argument('--num_prompts', default=3, type=int, help='number of inserted prompts for test-time adaptation.')    
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA')    
    parser.add_argument('--lambda_bp', default=30, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA-BP')    

    # compared method settings
    parser.add_argument('--margin_e0', default=0.4*math.log(1000), type=float, help='the entropy margin for sar')    

    # output settings
    parser.add_argument('--output', default='./outputs', help='the output directory of this experiment')
    parser.add_argument('--tag', default='_first_experiment', type=str, help='the tag of experiment')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # create logger for experiment
    args.output += '/' + args.algorithm + args.tag + '/'
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    logger.info(args)

    # configure the domains for adaptation
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    # create model
    if args.model == "resnet50_gn_timm":
        net = timm.create_model('resnet50_gn', pretrained=True)
    elif args.model == "vitbase_timm":
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
    else:
        assert False, NotImplementedError     
        
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    if args.algorithm == 'tent':
        net = tent.configure_model(net)
        params, _ = tent.collect_params(net)
        if args.model == 'vitbase_timm':
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        elif args.model == 'resnet50_gn_timm':
            optimizer = torch.optim.SGD(params, lr=0.00025, momentum=0.9)
        adapt_model = tent.Tent(net, optimizer)
    elif args.algorithm == 'actmad':
        net = PromptViT(net, 0).cuda()
        net = actmad.configure_model(net).cuda()
        params, _ = actmad.collect_params(net)
        if args.model == 'vitbase_timm':
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)
        elif args.model == 'resnet50_gn_timm':
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        adapt_model = actmad.ActMAD(net, optimizer, 30)
        train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'eata':
        # compute fisher informatrix
        fishers, _ = get_fisher(args, net)
        net = eata.configure_model(net)
        params, _ = eata.collect_params(net)
        if args.model == 'vitbase_timm':
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        elif args.model == 'resnet50_gn_timm':
            optimizer = torch.optim.SGD(params, lr=0.00025, momentum=0.9)
        adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'sar':
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)
        base_optimizer = torch.optim.SGD
        if args.model == 'vitbase_timm':
            optimizer = SAM(params, base_optimizer, lr=0.001, momentum=0.9)
        elif args.model == 'resnet50_gn_timm':
            optimizer = SAM(params, base_optimizer, lr=0.00025, momentum=0.9)
        # NOTE: set margin_e0 to 0.4*math.log(200) on ImageNet-R
        adapt_model = sar.SAR(net, optimizer, margin_e0=args.margin_e0)
    elif args.algorithm == 'deyo':
        net = deyo.configure_model(net)
        params, param_names = deyo.collect_params(net)
        if args.model == 'vitbase_timm':
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        elif args.model == 'resnet50_gn_timm':
            optimizer = torch.optim.SGD(params, lr=0.00025, momentum=0.9)
        adapt_model = deyo.DeYO(net, optimizer)
    elif args.algorithm == 'cotta':
        net = cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        if args.model == 'vitbase_timm':
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)
        elif args.model == 'resnet50_gn_timm':
            optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9)
        adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False)
    elif 'spa' in args.algorithm:
        from models.byol_wrapper import BYOLWrapper
        projector_dim = net.head.in_features if args.model == 'vitbase_timm' else net.fc.in_features
        net = BYOLWrapper(net, projector_dim=projector_dim).cuda()
        net = spa.configure_model(net)
        params, param_names = spa.collect_params(net)
        if args.model == 'vitbase_timm':
            optimizer = torch.optim.SGD([{'params': params, 'lr':0.01, 'momentum':0.9}, {'params': net.predictor.parameters(), 'lr':0.05 , 'momentum':0.9}])
            mask_ratio, noise_ratio = 0.2, 0.4        
        elif args.model == 'resnet50_gn_timm':
            optimizer = torch.optim.SGD([{'params': params, 'lr':0.0025, 'momentum':0.9}, {'params': net.predictor.parameters(), 'lr':0.025 , 'momentum':0.9}])
            mask_ratio, noise_ratio = 0.1, 0.1
        adapt_model = spa.SPA(net, optimizer, noise_ratio=noise_ratio, freq_mask_ratio=mask_ratio)
        use_spa_with_others(args, adapt_model) # spa+tent, spa+eta, spa+eata, spa+actmad, spa+tent+actmad
    elif args.algorithm == 'no_adapt':
        adapt_model = net
    else:
        assert False, NotImplementedError


    corrupt_acc, corrupt_ece = [], []
    for corrupt in corruptions:
        args.corruption = corrupt
        logger.info(args.corruption)

        if args.corruption == 'rendition':
            adapt_model.imagenet_mask = imagenet_r_mask
        elif args.corruption == 'adversial':
            adapt_model.imagenet_mask = imagenet_a_mask
        else:
            adapt_model.imagenet_mask = None

        val_dataset, val_loader = prepare_test_data(args)

        torch.cuda.empty_cache()
        top1, top5 = validate_adapt(val_loader, adapt_model, args)
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.6f} and Top-5 Accuracy: {top5:.6f}")
        corrupt_acc.append(top1)

        # reset model before adapting on the next domain
        if args.algorithm == 'no_adapt':
            continue
        else:
            adapt_model.reset()
        
        logger.info(f'mean acc of corruption: {sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0}')
        logger.info(f'corrupt acc list: {[_.item() for _ in corrupt_acc]}')
