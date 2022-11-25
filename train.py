#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import argparse
from trainer.trainer import train
from tools.config_loader import get_config


if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # when running code in the command line, parser makes the command line interactive:
    # if we then type "train.py --help" we get a summary of our options 
    
    parser = argparse.ArgumentParser(description='Settings.')  
    parser.add_argument('-n', '--exp_name', default='exp_name', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-d', '--dataset', default='Clotho', type=str,
                        help='Dataset used.')
    parser.add_argument('-w', '--word', default='True', type=str,
                        help='Pre-trained word embedding.')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-e', '--batch', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed')
    parser.add_argument('-k', '--keywords', default='True', type=str,
                        help='Use keywords or not.')

    args = parser.parse_args()

    config = get_config(args.config)

    # enables the user to update the parameters in the yaml file by specifying options when calling train.py
    
    config.exp_name = args.exp_name
    config.dataset = args.dataset
    config.data.batch_size = args.batch
    config.training.seed = args.seed  
    config.word_embedding.pretrained = eval(args.word)
    config.keywords = eval(args.keywords)
    train(config)
