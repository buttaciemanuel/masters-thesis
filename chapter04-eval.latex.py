'''
This file allows visualizing the results from the simulations producing the output in Latex style (if locally installed).

Examples
--------
The following command will visualize the simulations within the `./simulations` directory for the `FEMNIST` dataset, 
and it will group them by (balancedness, convexity) on the rows and by default metric (either loss or accuracy) on the columns.
>>> python chapter04-eval.py --input ./simulations --output ./visualizations --dataset femnist --groupby balancedness+convexity
'''

import argparse
import json
import matplotlib.pyplot as plt
import numpy
import os
import pandas as pd
import seaborn
import shutil

parser = argparse.ArgumentParser('visualize', usage = 'Visualize the curves from simulations.')
parser.add_argument('-i', '--input', type = str, default = './simulations')
parser.add_argument('-m', '--metric', type = str, choices = [ 'accuracy', 'loss', 'loss+accuracy' ], default = 'loss+accuracy')
parser.add_argument('-d', '--dataset', type = str, required = True, choices = [ 'cifar10', 'cifar100', 'femnist' ])
parser.add_argument('-g', '--groupby', type = str, required = True, choices = [ 'balancedness', 'convexity', 'balancedness+convexity', 'algorithm', 'epochs', 'max_gradient_norm', 'step_size' ])
parser.add_argument('-c', '--colorby', type = str, choices = [ 'balancedness', 'convexity', 'balancedness+convexity', 'algorithm', 'epochs', 'max_gradient_norm', 'step_size' ])
parser.add_argument('-p', '--partition', type = str, default = 'testing', choices = [ 'training', 'testing' ])
parser.add_argument('-y', '--same_y_scale', action = 'store_true', default = False)
parser.add_argument('-G', '--grid', action = 'store_true', default = False)
parser.add_argument('-o', '--output', type = str, default = './visualizations')

if shutil.which('latex'):
    params = {
        'ytick.color' : 'black',
        'xtick.color' : 'black',
        'text.usetex' : True,
        'font.family' : 'serif',
        'font.serif' : [ 'Times New Roman' ],
        'text.latex.preamble': '\\usepackage{times}',
        'legend.frameon': True,
    }
else:
    print('error: no latex detected to execute this script')
    exit(1)

palette = 'Set1'

plt.rcParams.update(params)

if __name__ == '__main__':
    args = parser.parse_args()
    frame = []

    for file in os.listdir(args.input):
        with open(os.path.join(args.input, file), 'r') as file:
            content = json.load(file)
            algorithm = r'\textsc{FedAvg}'

            if content['algorithm'] == 'fedprox':
                algorithm = r'\textsc{FedProx} ($\alpha = ' + str(content['alpha']) + '$)'
            elif content['algorithm'] == 'ours':
                algorithm = r'\textrm{Ours} ($\beta = ' + str(content['beta']) + '$)'

            convexity = 'Nonconvex' if content['nonconvex'] else 'Convex'
            balancedness = 'Balanced' if content['class_imbalance'] == 0 and content['data_imbalance'] == 0 else 'Imbalanced'
            table = [
                {
                    'algorithm': algorithm,
                    'dataset': content['dataset'],
                    'epochs': '$E = {:2}$'.format(content['n_epochs']),
                    'scheme': content['weighting_scheme'],
                    'convexity': convexity,
                    'balancedness': balancedness,
                    'balancedness+convexity': '{}, {}'.format(balancedness, convexity),
                    'max_gradient_norm': '$G = {:6}$'.format(content['max_gradient_norm'] if numpy.isfinite(content['max_gradient_norm']) else r'\infty'),
                    'step_size': '$\gamma = {}$'.format(content['step_size']),
                    'round': datapoint['step'],
                    'metric': 'loss',
                    'value': datapoint['loss.{}'.format(args.partition)],
                }
                for datapoint in content['metrics']
            ]

            table.extend([
                {
                    'algorithm': algorithm,
                    'dataset': content['dataset'],
                    'epochs': '$E = {:2}$'.format(content['n_epochs']),
                    'scheme': content['weighting_scheme'],
                    'convexity': convexity,
                    'balancedness': balancedness,
                    'balancedness+convexity': '{}, {}'.format(balancedness, convexity),
                    'max_gradient_norm': '$G = {:6}$'.format(content['max_gradient_norm'] if numpy.isfinite(content['max_gradient_norm']) else r'\infty'),
                    'step_size': '$\gamma = {}$'.format(content['step_size']),
                    'round': datapoint['step'],
                    'metric': 'accuracy',
                    'value': datapoint['accuracy.{}'.format(args.partition)],
                }
                for datapoint in content['metrics']
            ])

            frame.extend(table)

    frame = pd.DataFrame(frame).sort_values(by = [ 'scheme', 'dataset', 'balancedness', 'convexity', 'max_gradient_norm', 'step_size', 'algorithm', 'round' ])
    
    if args.dataset not in frame.dataset.unique():
        print('error: dataset {} not present in the simulations'.format(args.dataset))
        exit(1)

    frame = frame[frame.dataset == args.dataset]

    if args.metric != 'loss+accuracy':
        frame = frame[frame.metric == args.metric]
        height = 2.75
        aspect = 0.95 if not args.same_y_scale else 0.85
        
    else:
        frame = frame.sort_values(by = 'metric', ascending = False)
        height = 2.75
        aspect = 1.1 if not args.same_y_scale else 1

    fgrid = seaborn.relplot(frame, x = 'round', y = 'value', hue = args.colorby, style = args.colorby, kind = 'line', row = 'metric', col = args.groupby, height = height, aspect = aspect, col_order = sorted(frame[args.groupby].unique().tolist()), hue_order = sorted(frame[args.colorby].unique().tolist()) if args.colorby is not None else None, palette = palette, facet_kws = { 'margin_titles': True, 'sharex': False, 'sharey': 'row' if args.same_y_scale else False, 'despine': False })
    
    fgrid.set_xlabels(size = 14)
    fgrid.set_ylabels(size = 14)
    fgrid.tick_params(labelsize = 14)

    if args.metric == 'loss+accuracy':
        fgrid.axes[0, 0].set_ylabel('loss')
        fgrid.axes[1, 0].set_ylabel('accuracy')
    else:
       fgrid.axes[0, 0].set_ylabel(args.metric) 

    fgrid.set_titles(col_template = '{col_name}', row_template = '', size = 14)
    
    if args.colorby is not None:
        fgrid.legend.set_title('')
        plt.setp(fgrid._legend.get_texts(), fontsize = 14)

    if args.grid:
        for ax in fgrid.axes.ravel():
            ax.grid(True)

    fgrid.tight_layout()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    fgrid.savefig(os.path.join(args.output, '{}:groupby={}:partition={}:scale={}:metric={}.pdf'.format(args.dataset, args.groupby, args.partition, args.same_y_scale, args.metric)), dpi = 300, bbox_inches = 'tight')
         
    plt.show()