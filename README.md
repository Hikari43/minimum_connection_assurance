# Minimum Connection Assurance (MiCA)

## Installation
```
conda env create -f requirement.yml
```

## Code
This code is based on the code of ["Pruning neural networks without any data by iteratively conserving synaptic flow"](https://arxiv.org/abs/2006.05467).
[(code)](https://github.com/ganguli-lab/Synaptic-Flow)

### Experiments
`Configs` folder includes config files.

Experiment by executing the following commands:
* If you want to experiment more than once:
    ```
    ./experiment_runner.sh \
    [config file] \
    [experimental count] \
    [pruner name (use if you want to use a pruning mask from another model)] \
    [experiment name (use if you want to use a pruning mask from another model)] \
    [compression ratio list (10**x)]
    ```
* If you want to experiment once:
    ```
    python main.py \
    --config [config file] \
    --run-number [experimental number] \
    --seed [experimental seed] \
    --compression [compression ratio (10**x)] \
    --mask-file [model file (use this option if you want to use a pruning mask from another model)]
    ```

### Examples

* Experiment with MiCA-IGQ on CIFAR-10 using ResNet-20 three times each at compression ratios of $10^{1.0}$, $10^{1.5}$, and $10^{2.0}$:
    ```
    ./experiment_runner.sh \
    Configs/singleshot/mica/igq/cifar10_lottery_resnet20/cifar10_lottery_resnet20_rand.yml \
    3 \
    None \
    None \
    1.0 1.5 2.0
    ```

* Experiment with MiCA-IGQ on CIFAR-10 using VGG-16 three times each at corrected compression ratios of $10^{1.0}$, $10^{1.5}$, and $10^{2.0}$ (NOTE: you must experiment with RPI-IGQ beforehand):
    ```
    ./experiment_runner_effective_comp.sh \
    Configs/singleshot/mica/igq/cifar10_lottery_vgg16_bn/cifar10_lottery_vgg16_bn_rand.yml \
    3 \
    mica_igq \
    cifar10_lottery_vgg16_bn_shuffle \
    1.0 1.5 2.0
    ```

    (When using experiment_runner_effective_comp.sh, the model derived from the effective sparsity must have been obtained beforehand by `running Graphs/get_model_effective_comp.ipynb`)

## Plots 

You can visualize the results of experiments to use notebooks in the `Graphs` folder.