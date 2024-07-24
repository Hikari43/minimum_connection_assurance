from tqdm import tqdm

def prune_loop( model, loss, pruner, dataloader, device, density, schedule, scope, epochs,
                reinitialize=False, train_mode=False, shuffle=False, invert=False, 
                shuffle_vertices=False, lr=None, sparsity_distribution=None, 
                mask_file=None, V_out_select='max', set_only_V_in=False):
    r"""Applies score mask loop iteratively to a final density level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(
            model=model, loss=loss, dataloader=dataloader, device=device, 
            density=density, scope=scope, lr=lr, sparsity_distribution=sparsity_distribution, 
            mask_file=mask_file, V_out_select=V_out_select, set_only_V_in=set_only_V_in)
        if schedule == 'exponential':
            dense = density**((epoch + 1) / epochs)
        elif schedule == 'linear':
            dense = 1.0 - (1.0 - density)*((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(dense, scope)
    
    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle(shuffle_vertices)



    #     # Confirm sparsity level
    # remaining_params, total_params = pruner.stats()
    # if np.abs(remaining_params - total_params*density) >= 5:
    #     print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*density))
    #     quit()