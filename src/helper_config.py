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
        'valid': ('07-01-2022', '06-30-2022'),
        'test' : ('07-01-2022', '12-31-2023')
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
        }
    }
    
    ### Features and tickers
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    tickers  = ['ATOM', 'AVAX', 'BNB', 'BTC', 'ETC', 'ETH', 'LINK', 'LTC', 'SOL', 'XMR']
    max_len  = 64
    n_forward = 1
    
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