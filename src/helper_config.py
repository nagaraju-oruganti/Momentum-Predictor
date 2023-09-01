import random, os
import numpy as np
import torch

### SEED EVERYTHING
def seed_everything(seed: int = 42):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    print(f'set seed to {seed}')

class Config:
    
    # random seed
    seed = 42
    seed_everything(seed = seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Repositories
    data_dir = ''
    models_dir = ''
    data_dest_dir = ''
    
    sample_run = False
    
    ### Train, valid, and test splits
    splits = {
        'train': ('01-01-2021', '06-30-2022'),
        #'valid': ('07-01-2022', '06-30-2022'),
        #'test' : ('07-01-2022', '12-31-2023')
    }
    
    folds = {
        1 : {
            'train': ('01-01-2021', '09-30-2021'),
            'valid': ('10-01-2021', '12-31-2021')
            },
        2 : {
            'train': ('01-01-2021', '12-31-2021'),
            'valid': ('01-01-2022', '03-31-2022')
        },
        3 : {
            'train': ('01-01-2021', '03-31-2022'),
            'valid': ('04-01-2022', '06-30-2022')
        },
        
        ### THIS WILL BE USED IN SIMULATION
        1000: {
            'train': ('01-01-2021', '12-31-2021'),      # small window to save time
            'valid': ('01-01-2021', '12-31-2023'),
        },
    }
    
    ### Features and tickers
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    tickers  = ['ATOM', 'AVAX', 'BNB', 'BTC', 'ETC', 'ETH', 'LINK', 'LTC', 'SOL', 'XMR']
    model_kind = 'hslstm'
    max_len  = 64
    n_forward = 1
    num_heads = 4
    num_layers = 2
    train_on_rmse = False
    fine_tune = False
    dropout_prob = 0
    
    ### Gramian Angular Field
    params = dict(
        image_size      = 64,
        method          = 'difference',
        sample_range    = (0, 1)
    )
    
    ### DATA LABELS
    long_period = 26
    short_period = 12
    signal_period = 9
    prep_for_train = True
    
    dict_volume_threshold = {
        'ATOM'  : 1479932.0,
        'AVAX'  : 1360353.5,
        'BNB'   : 365430.5,
        'BTC'   : 48171.0,
        'ETC'   : 2009940.0,
        'ETH'   : 417021.5,
        'LINK'  : 2831656.5,
        'LTC'   : 393902.5,
        'SOL'   : 3406926.0,
        'XMR'   : 32844.5
        }
    
    # Batch sizes
    train_batch_size = 16
    valid_batch_size = 32
    
    # train with batches    (better way to manage memory with training on batches at the expense of clock time)
    train_batches = 1       # default is 1 - means batching will have no effect
    batch_indices = [200, 400, 600, 800, 1000]
    batch_dir = None        # save batches of dataset to the local directory and load which training
    
    #### PRETRAIN CONFIGURATION
    objective_pretrain = False
    pretrain_config = dict(
        data_dir                    = '/Volumes/Oruganti',
        reverse_sample              = 0.5,
        hourly                      = False,
        daily                       = True,
        daily_vol_thresh_file       = 'pretrain_daily_volume_thresholds.pkl',
        hourly_vol_thresh_file      = 'pretrain_hourly_volume_thresholds.pkl',
        daily_suffix_for_ticker     = 'd',
        hourly_suffix_for_ticker    = 'h',
        load_weights_from           = None,
        n_train_tickers             = 400,
        n_valid_tickers             = 200,
    )
    
    pretrain_folds = {
        1 : {
            'train': ('01-01-2000', '12-31-2019'),
            'valid': ('01-01-2020', '12-31-2023')
        },
    }
    
    ignore_macd = True
    
    ## SIMULATION PARAMS
    sim_config = dict(
        initial_balance = 1_000_000,
        buy_txn_cost    = 0.05/100.,
        sell_txn_cost   = 0.03/100.,
        load_weights_from = None,
    )