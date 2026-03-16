"""
This file runs the main training/val loop
"""

import json
import os
import pprint
import sys

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
    opts = TrainOptions().parse()
    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    coach = Coach(opts)
    if opts.job == 'Inversion':
        coach.invert_clip()
    elif opts.job == 'Manipulate':
        coach.manipulate_clip_one()
    elif opts.job == 'Comparision':
        coach.invert_clip_one()
    elif opts.job == 'Visualization':
        coach.umap_visualization()
    else:
        raise ValueError(
            'No job to choose, please choose Inversion or Manipulate!'
        )


if __name__ == '__main__':
    main()
