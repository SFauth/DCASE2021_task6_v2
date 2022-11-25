#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

# this script just creates a function get_config, which reads in the .yaml file specified in get_config. 
# The .yaml file contains all relevant (hyper) parameters that are used to train the model.
# E.g. batch_size, activation and layer size of the decoder, ...

import yaml
from dotmap import DotMap


def get_config(config_name='settings'):

    with open('settings/{}.yaml'.format(config_name), 'r') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    return config
