import yaml
import easydict
import os
import argparse

parser = argparse.ArgumentParser(description='Code for *Semantic Segmentation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='cityscapes.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file))
args = easydict.EasyDict(args)



