def get_default_params():
    params = dict()

    params['wandb_project'] = 'FCD'
    params['model_type'] = 'MS_DSA_NET'
    params["model_returns_vaeloss"] = False # this will be automatically assigned in get_model
    params['sa_type'] = 'parallel'
    params['feature_size'] = 16
    params['project_size'] = 64  # DSA projection size
    params['patch_size'] = 128

    params['chans_in'] = 2
    params['chans_out'] = 2
    params['seq'] = 't1_reg+flair_reg'  #input sequence file names separated by '+', e.g. 't1', 'flair', 't1+flair', 't1_reg+flair_reg+curv+area' #filenames should end with .nii.gz

    params['num_workers'] = 4
    params['samples_per_case'] = 4
    params['batch_size'] = 1
    params['gradient_accumulation_steps'] = 1
    params['use_amp'] = True
    params['adjust_lr_with_batch_size'] = False


    params['min_region_size'] = 50  # -1 for keeping the largest connected component only

    params['deterministic'] = 'seed_only' # 'off', 'seed_only', 'strict'
    params['seed'] = 42  # random seed for reproducibility

    params['lr'] = 1e-4
    params['weight_decay'] = 1e-5
    params['min_lr'] = 1e-6
    params['max_epochs'] = 300
    params['min_epochs'] = 120
    params['warmup_epochs'] = 10
    params['early_stopping_patience'] = 25
    params['val_loss_ema_alpha'] = 0.7 # 0 for not using EMA (val_loss_ema = ( 1 - alpha) * val_loss + alpha * val_loss_ema)

    params['loss'] = 'DiceLoss'
    params['lambda_dice'] = 1.0
    params['lambda_ce'] = 1.0
    params['lambda_focal'] = 1.0
    params['ce_background_weight'] = 0.5
    params['ce_fcd_weight'] = 0.5
    params['gamma_focal'] = 2.0
    params['gdice_wtype'] = "square"  # 'square', 'simple', 'uniform'
    params['jaccard'] = False
    params['square_pred'] = False
    params['sigmoid'] = False
    params['softmax'] = True
   

    params['coarse_dropout_max_prob'] = 0.0  # maximum dropout probability for coarse dropout
    params['coarse_dropout_start_epoch'] = 0.0 # start epoch for coarse dropout
    params['gridmask_max_prob'] = 0.0  # maximum dropout probability for gridmask
    params['gridmask_start_epoch'] = 0.0  # start epoch for gridmask

    params['segresnet_upsample_mode'] = 'pixelshuffle'  # used in all SegResNet based architectures, options: nontrainable, deconv, pixelshuffle
    params['segresnet_deeper'] = False  # used in all SegResNet based architectures, whether to use a deeper architecture
 
    params['tv_loss_norm'] = 'l1' # 'l1' or 'l2'
    params['tv_loss_weight'] = 0.0  # 0.1
    params['tvloss_exclude_borders'] = False  # whether to exclude borders from TV loss computation
    params["boundaryloss_weight"] = 0.0 # 0.3

    params['loss_vae_weight'] = 0.2
    
    params['keep_latest_model'] = False

    return params