import argparse
import json
import os
from Experiments import singleshot

def get_config(args):
    import sys
    import yaml
    def argv_to_vars(argv):
        var_names = []
        for arg in argv:
            if arg.startswith("-") and arg_to_varname(arg) != "config":
                var_names.append(arg_to_varname(arg))
        return var_names
    
    def arg_to_varname(st: str):
        st = trim_preceding_hyphens(st)
        st = st.replace("-", "_")
        return st.split("=")[0]
    
    def trim_preceding_hyphens(st):
        i = 0
        while st[i] == "-":
            i += 1
        return st[i:]

    # get commands from command line
    override_args = argv_to_vars(sys.argv)
    # load yaml file
    with open(args.config) as f:
        loaded_yaml = yaml.safe_load(f)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)
    print(f'-----Element further overwritten from the config-----')
    for a in override_args:
        print(a)
    print()
    
def show_args(args):
    print('-----Exp setting-----')
    for k, v in args.__dict__.items():
        print(f'{k} : {v}')
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network Compression')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','tiny-imagenet','imagenet'],
                        help='dataset (default: mnist)')
    training_args.add_argument('--model', type=str, default='fc', choices=['fc','conv',
                        'vgg11','vgg11-bn','vgg13','vgg13-bn','vgg16','vgg16-bn','vgg19','vgg19-bn',
                        'resnet18','resnet20','resnet32','resnet34','resnet44','resnet50',
                        'resnet56','resnet101','resnet110','resnet110','resnet152','resnet1202',
                        'wide-resnet18','wide-resnet20','wide-resnet32','wide-resnet34','wide-resnet44','wide-resnet50',
                        'wide-resnet56','wide-resnet101','wide-resnet110','wide-resnet110','wide-resnet152','wide-resnet1202'],
                        help='model architecture (default: fc)')
    training_args.add_argument('--model-class', type=str, default='default', choices=['default','lottery','tinyimagenet','imagenet'],
                        help='model class (default: default)')
    training_args.add_argument('--dense-classifier', type=bool, default=False,
                        help='ensure last layer of model is dense (default: False)')
    training_args.add_argument('--use-skip-connection', type=bool, default=True,
                        help='use skip connection (for ResNet)')
    training_args.add_argument('--pretrained', type=bool, default=False,
                        help='load pretrained weights (default: False)')
    training_args.add_argument('--optimizer', type=str, default='adam', choices=['sgd','momentum','adam','rms'],
                        help='optimizer (default: adam)')
    training_args.add_argument('--train-batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    training_args.add_argument('--test-batch-size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    training_args.add_argument('--pre-epochs', type=int, default=0,
                        help='number of epochs to train before pruning (default: 0)')
    training_args.add_argument('--post-epochs', type=int, default=10,
                        help='number of epochs to train after pruning (default: 10)')
    training_args.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    training_args.add_argument('--lr-drops', type=int, nargs='*', default=[],
                        help='list of learning rate drops (default: [])')
    training_args.add_argument('--lr-drop-rate', type=float, default=0.1,
                        help='multiplicative factor of learning rate drop (default: 0.1)')
    training_args.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    training_args.add_argument('--data-parallel', type=bool, default=False,
                        help='use huggingface accelerator (default: False)')
    # Pruning Hyperparameters
    pruning_args = parser.add_argument_group('pruning')
    pruning_args.add_argument('--pruner', type=str, default='rand', 
                        choices=['rand', 'mag', 'snip', 'grasp', 'synflow', 'ep', 'mica'],
                        help='prune strategy (default: rand)')
    pruning_args.add_argument('--sparsity-distribution', type=str, default='erk',
                        choices=['snip', 'grasp', 'synflow', 'erk', 'igq', 'epl'])
    pruning_args.add_argument('--compression', type=float, default=0.0,
                        help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
    pruning_args.add_argument('--prune-epochs', type=int, default=1,
                        help='number of iterations for scoring (default: 1)')
    pruning_args.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                        help='whether to use a linear or exponential compression schedule (default: exponential)')
    pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global','local'],
                        help='masking scope (global or layer) (default: global)')
    pruning_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                        help='ratio of prune dataset size and number of classes (default: 10)')
    pruning_args.add_argument('--prune-batch-size', type=int, default=256,
                        help='input batch size for pruning (default: 256)')
    pruning_args.add_argument('--prune-bias', type=bool, default=False,
                        help='whether to prune bias parameters (default: False)')
    pruning_args.add_argument('--prune-batchnorm', type=bool, default=False,
                        help='whether to prune batchnorm layers (default: False)')
    pruning_args.add_argument('--prune-residual', type=bool, default=False,
                        help='whether to prune residual connections (default: False)')
    pruning_args.add_argument('--prune-train-mode', type=bool, default=False,
                        help='whether to prune in train mode (default: False)')
    pruning_args.add_argument('--reinitialize', type=bool, default=False,
                        help='whether to reinitialize weight parameters after pruning (default: False)')
    pruning_args.add_argument('--shuffle', type=bool, default=False,
                        help='whether to shuffle masks after pruning (default: False)')
    pruning_args.add_argument('--shuffle-vertices', type=bool, default=False,
                        help='whether to shuffle vertices after pruning (default: False)')
    pruning_args.add_argument('--invert', type=bool, default=False,
                        help='whether to invert scores during pruning (default: False)')
    pruning_args.add_argument('--pruner-list', type=str, nargs='*', default=[],
                        help='list of pruning strategies for singleshot (default: [])')
    pruning_args.add_argument('--prune-epoch-list', type=int, nargs='*', default=[],
                        help='list of prune epochs for singleshot (default: [])')
    pruning_args.add_argument('--compression-list', type=float, nargs='*', default=[],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
    pruning_args.add_argument('--level-list', type=int, nargs='*', default=[],
                        help='list of number of prune-train cycles (levels) for multishot (default: [])')
    pruning_args.add_argument('--mask-file', type=str, default=None)
    pruning_args.add_argument('--V-out-select', type=str, default='max',
                        help='select V_out settings (default : max)')
    pruning_args.add_argument('--set-only-V-in', type=bool, default=False)
    ## Experiment Hyperparameters ##
    parser.add_argument('--experiment', type=str, default='singleshot', 
                        choices=['singleshot','multishot','unit-conservation',
                        'layer-conservation','imp-conservation','schedule-conservation'],
                        help='experiment name (default: example)')
    parser.add_argument('--expid', type=str, default='',
                        help='name used to save results (default: "")')
    parser.add_argument('--result-dir', type=str, default='Results/data',
                        help='path to directory to save results (default: "Results/data")')
    parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
    parser.add_argument('--workers', type=int, default='4',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--verbose', action='store_true',
                        help='print statistics during training and testing')
    parser.add_argument('--config',
                        default = None,
                        help='config file to use')
    parser.add_argument('--run-number',
                        default = 0,
                        help = 'experiment number for multiple runs')
    parser.add_argument('--dataset-root', type=str, default=None)
    args = parser.parse_args()
    if args.config != None:
        get_config(args)

    show_args(args)

    ## Construct Result Directory ##
    if args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        pruner_names = {'rand': 'rand', 'snip': 'cs', 'grasp': 'grasp', 
                        'synflow': 'sf', 'erk' : 'erk', 'igq' : 'igq'}
        pruner_names['mica'] = f'mica_{pruner_names[args.sparsity_distribution]}'
        if args.experiment == 'singleshot':
            result_dir   =  f'{args.result_dir}/{args.experiment}/{args.expid}/' \
                            f'{pruner_names[args.pruner]}-{args.compression}-{args.prune_epochs}/run_{args.run_number}'
        setattr(args, 'save', True)
        setattr(args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            val = ""
            while val not in ['yes', 'no']:
                val = input(f"{result_dir} exists.  Overwrite (yes/no)? ")
            if val == 'no':
                quit()

    ## Save Args ##
    if args.save:
        with open(args.result_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)

    ## Run Experiment ##
    if args.experiment == 'singleshot':
        singleshot.run(args)




