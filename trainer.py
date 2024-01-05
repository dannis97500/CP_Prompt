import copy
import logging
import os
import os.path
import sys
import time
from utils.toolkit import  accuracy_domain

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import shutil 

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        if(args["prefix"]=="prefix_one_prompt"):
            _prefix_prompt_train(args)
            return
        else:
            _train(args)
           
    myseed = 42069  


    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)# # sets the seed for generating random numbers
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)# # Sets the seed for generating random numbers on all GPUs.
       
def _prefix_prompt_train(args):
    logfilename = './logs/{}_{}_{}_{}_'.format(args['model_name'],args['query_type'],
                                                args['dataset'], args['init_cls'])+ time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())                                         
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    os.makedirs(logfilename)
    print(logfilename)
    _set_random()
    _set_device(args)
    print_args(args)

    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = factory.get_model(args['model_name'], args)
    cnn_curve, nme_curve = {'top1': []}, {'top1': []}

    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.begin_incremental(data_manager)
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['grouped']['total'])
            nme_curve['top1'].append(nme_accy['top1'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
        else:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['grouped']['total'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
    torch.save(model, os.path.join(logfilename, "task_{}.pth".format(int(task))))

  
def _evaluate(model,y_pred, y_true):
    ret = {}
    grouped = accuracy_domain(y_pred.T[0], y_true, model._known_classes, class_num=model.class_num)
    ret['grouped'] = grouped
    ret['top1'] = grouped['total']
    return ret

def _train(args):
    logfilename = './logs/{}_{}_{}_{}_{}_{}_{}_'.format(args['prefix'], args['seed'], args['model_name'],args['net_type'],
                                                args['dataset'], args['init_cls'], args['increment'])+ time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())                                      
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    os.makedirs(logfilename)
    print(logfilename)
    _set_random()
    _set_device(args)
    print_args(args)

    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = factory.get_model(args['model_name'], args)
    cnn_curve, nme_curve = {'top1': []}, {'top1': []}


    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.begin_incremental(data_manager)

        model.incremental_train(data_manager)
        
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['top1'])
            nme_curve['top1'].append(nme_accy['top1'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
        else:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['top1'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
    torch.save(model, os.path.join(logfilename, "task_{}.pth".format(int(task))))

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1) 
    torch.cuda.manual_seed(1) 
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
