import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = load.device(args.gpu)

    ## Data ##
    print(f'Loading {args.dataset} dataset.')
    input_shape, num_classes = load.dimension(args.dataset) 

    train_loader, val_loader, test_loader, prune_loader = load.get_dataloaders(
        dataset_name=args.dataset, batch_size=args.train_batch_size, prune_batch_size=args.prune_batch_size, 
        workers=args.workers, seed=args.seed, length=args.prune_dataset_ratio * num_classes, root=args.dataset_root
    )

    # prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    # train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    # test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

    ## Model, Loss, Optimizer ##
    print(f'Creating {args.model_class}-{args.model} model.')
    model = load.model(args.model, args.model_class)(input_shape, 
                                                    num_classes, 
                                                    args.dense_classifier, 
                                                    args.pretrained,
                                                    args.use_skip_connection).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Save Original ##
    torch.save(model.state_dict(),     f"{args.result_dir}/model_orig.pt")
    torch.save(optimizer.state_dict(), f"{args.result_dir}/optimizer_orig.pt")
    torch.save(scheduler.state_dict(), f"{args.result_dir}/scheduler_orig.pt")

    ## Pre-Train ##
    print(f'Pre-Train for {args.pre_epochs} epochs.')
    assert args.pre_epochs == 0
    pre_result = train_eval_loop(
        model=model, loss=loss, optimizer=optimizer, scheduler=scheduler, 
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
        device=device, epochs=args.pre_epochs, verbose=args.verbose)

    ## Prune ##
    print(f'Pruning with {args.pruner} for {args.prune_epochs} epochs.')
    pruner  = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    density = 10**(-float(args.compression))
    prune_loop( model=model, loss=loss, pruner=pruner, dataloader=prune_loader, device=device, density=density, 
                schedule=args.compression_schedule, scope=args.mask_scope, epochs=args.prune_epochs, reinitialize=args.reinitialize, 
                train_mode=args.prune_train_mode, shuffle=args.shuffle, invert=args.invert, shuffle_vertices=args.shuffle_vertices, 
                lr=args.lr, sparsity_distribution=args.sparsity_distribution, mask_file=args.mask_file, V_out_select=args.V_out_select, 
                set_only_V_in=args.set_only_V_in)
    torch.save(model.state_dict(),     f"{args.result_dir}/model_after_pruning.pt")

    if args.data_parallel:
        gpu_count = torch.cuda.device_count()
        print(f'# of GPUs : {gpu_count}')
        assert args.train_batch_size % gpu_count == 0
        assert args.test_batch_size  % gpu_count == 0
        model = torch.nn.DataParallel(model)
        loss = nn.CrossEntropyLoss()
        opt_class, opt_kwargs = load.optimizer(args.optimizer)
        optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Post-Train ##
    print(f'Post-Training for {args.post_epochs} epochs.')
    post_result = train_eval_loop(
        model=model, loss=loss, optimizer=optimizer, scheduler=scheduler, 
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
        device=device, epochs=args.post_epochs, verbose=args.verbose) 

    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
    prune_result = metrics.summary( model, 
                                    pruner.scores,
                                    metrics.flop(model, input_shape, device),
                                    lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
    total_params = int((prune_result['density'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    total_flops = int((prune_result['density'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print(f"Parameter density: {total_params}/{possible_params} ({total_params / possible_params:.4f})")
    print(f"FLOP density: {total_flops}/{possible_flops} ({total_flops / possible_flops:.4f})")

    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        if args.data_parallel:
            # model = accelerator.unwrap_model(model)
            # if accelerator.is_local_main_process:
            pre_result.to_pickle(f"{args.result_dir}/pre-train.pkl")
            post_result.to_pickle(f"{args.result_dir}/post-train.pkl")
            prune_result.to_pickle(f"{args.result_dir}/compression.pkl")
            torch.save(model.module.state_dict(),f"{args.result_dir}/model.pt")
            torch.save(optimizer.state_dict(),f"{args.result_dir}/optimizer.pt")
            torch.save(scheduler.state_dict(),f"{args.result_dir}/scheduler.pt")
        else:
            pre_result.to_pickle(f"{args.result_dir}/pre-train.pkl")
            post_result.to_pickle(f"{args.result_dir}/post-train.pkl")
            prune_result.to_pickle(f"{args.result_dir}/compression.pkl")
            torch.save(model.state_dict(),f"{args.result_dir}/model.pt")
            torch.save(optimizer.state_dict(),f"{args.result_dir}/optimizer.pt")
            torch.save(scheduler.state_dict(),f"{args.result_dir}/scheduler.pt")