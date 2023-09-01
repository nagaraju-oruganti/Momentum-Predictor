import os
import pandas as pd
import numpy as np
import pickle
from pyts.image import GramianAngularField
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

try:
    import talib
except:
    pass

import warnings
warnings.filterwarnings('ignore')

from scipy.ndimage import gaussian_filter
import gc

# VOLUME BARS
class VolumeBars:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.load_data()
        
        self.long_period = self.config.long_period
        self.short_period = self.config.short_period
        self.signal_period = self.config.signal_period
    
    def load_data(self):
        if self.config.objective_pretrain:
            self.df = self.load_data_for_pretrain()
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'], utc = True)
            
            self.load_vol_threshold_for_pretrain()
            self.config.tickers = self.df['tic'].unique().tolist()[:1000]
            self.config.splits['train'] = (self.df['Datetime'].min(), self.df['Datetime'].max())
        else:
            self.df = pd.read_csv(f'{self.data_dir}/long_TICKERS.csv')
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
            
    def load_data_for_pretrain(self):
        dfs = []
        data_dict = {
            'daily' : self.config.pretrain_config['daily'],
            'hourly' : self.config.pretrain_config['hourly'],
        }
        for kind, include in data_dict.items():
            if include:
                source_path = self.config.pretrain_config['data_dir'] + f'/{kind}_data.pqt'
                df = pd.read_parquet(source_path)
                df.reset_index(inplace = True, drop = False)
                df.rename(columns = {'Date' : 'Datetime', 'index' : 'Datetime', 'ticker' : 'tic'}, inplace = True)
                df['tic'] = df['tic'].apply(lambda x: f'{kind[0]}_{x}')
                dfs.append(df)
        
        df = pd.concat(dfs)
        return df
    
    def load_vol_threshold_for_pretrain(self):
        thresholds = {}
        for kind in ['daily', 'hourly']:
            with open(self.config.data_dir + f'/pretrain_{kind}_volume_thresholds.pkl', 'rb') as f:
                dict_thresh = pickle.load(f)
                dict_thresh = {f'{kind[0]}_{tic}' : v for tic, v in dict_thresh.items()}
                thresholds.update(dict_thresh)
        
        self.config.dict_volume_threshold = thresholds
        
    def make_one_ticker(self, ticker):
        volume_threshold = int(self.config.dict_volume_threshold[ticker])
        tick_df = self.df[self.df['tic'] == ticker][['Datetime'] + self.config.features]
        if not self.config.ignore_macd:
            macd_line, signal_line, macd_histogram = talib.MACD(tick_df['Close'], 
                                                                fastperiod = self.short_period,
                                                                slowperiod = self.long_period,
                                                                signalperiod = self.signal_period)
            tick_df['macd'] = macd_histogram
        else:
            tick_df['macd'] = 0
            
        start_condition = tick_df['Datetime'] >= self.config.splits['train'][0]
        end_condition   = tick_df['Datetime']  < self.config.splits['train'][1]
        df = tick_df[start_condition & end_condition]
        df.reset_index(inplace = True, drop = True)
        
        df.sort_values(by = 'Datetime', ascending = self.config.prep_for_train, inplace = True)
        df.dropna(subset = ['macd'], inplace = True)
        
        volume_bars = []
        current_volume = 0
        seq_start   = None
        open_price  = None
        high_price  = None
        low_price   = None
        close_price = None
        seq_end     = None
        end_idx     = None
        num_ts      = None         # num timestamps
        
        for _, row in df.iterrows():
            timestamp   = row['Datetime']
            _open       = row['Open']
            _high       = row['High']
            _low        = row['Low']
            _close      = row['Close']
            volume      = row['Volume']
            
            if seq_start == None:
                seq_start   = timestamp
                open_price  = _open
                close_price = _close
                high_price  = _high
                low_price   = _low
                num_ts      = 0
                
            # high, low
            high_price = max(high_price, _high)
            low_price = min(low_price, _low)
            
            # update volume
            current_volume += row['Volume']
            num_ts += 1 
            if current_volume >= volume_threshold:
                seq_end = timestamp
                
                # extact next 
                ### The following %change only works for training only. (anyway I am not using in the training or in the inference)
                D = tick_df[tick_df['Datetime'] > seq_end].head(10)
                dict_chg = {f'delta_{idx}': ((r['Close']/_close) - 1) for idx, (_, r) in enumerate(D.iterrows(), start = 1)}
                
                item = {
                    'ticker'    : ticker,
                    'start'     : seq_start if self.config.prep_for_train else seq_end,
                    'open'      : open_price if self.config.prep_for_train else _open,
                    'low'       : low_price,
                    'high'      : high_price,
                    'close'     : _close if self.config.prep_for_train else close_price,
                    'volume'    : current_volume,
                    'end'       : seq_end if self.config.prep_for_train else seq_start,
                    'num_timestamps': num_ts,
                }
                item.update(dict_chg)
                item.update({
                    'sum_macd': np.sum(D['macd']) if not self.config.ignore_macd else 0,
                    'mu_macd' : np.mean(D['macd']) if not self.config.ignore_macd else 0,
                    'dispersion_macd': np.std(D['macd']) / np.mean(D['macd']) if not self.config.ignore_macd else 0
                    })
                volume_bars.append(item)
                
                # reset
                current_volume = 0
                seq_start   = None
                open_price  = None
                high_price  = None
                low_price   = None
                close_price = None
                seq_end     = None
                num_ts      = None         # num timestamps
        
        df_vols = pd.DataFrame(volume_bars).sort_values(by = 'start')
        return df_vols

    def make(self):
        dfs = []
        que = tqdm(self.config.tickers, total = len(self.config.tickers), desc = 'Preprocessing')
        for tick in que:
            df = self.make_one_ticker(tick)
            
            # normalize
            target_cols = ['sum_macd', 'dispersion_macd']
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(df[target_cols])
            df[target_cols] = scaler.transform(df[target_cols])
            
            dfs.append(df)
            
            que.set_postfix({'ticker': tick})
            
        df = pd.concat(dfs)

        return df

class Preprocesser:
    def __init__(self, config):
        self.config = config
        self.normalizers = {}
        self.load_volume_bars()
        
    def load_volume_bars(self):
        if self.config.prep_for_train:            
            path = os.path.join(self.config.data_dir, 
                                'pretrain_volume_bars.pqt' if self.config.objective_pretrain else 'train_volume_bars.pqt')
        else:
            path = os.path.join(self.config.data_dir, 'infer_volume_bars.pqt')
            
        if os.path.exists(path):
            self.df = pd.read_parquet(path)
        else:
            self.df = VolumeBars(self.config).make()
            self.df.to_parquet(path)
            
    def normalize(self):
        cols = ['open', 'high', 'low', 'close']#, 'volume', 'num_timestamps']
        dfs = []
        for t in self.config.tickers:
            mm_scaler = MinMaxScaler()
            df = self.df[self.df['ticker'] == t]
            mm_scaler.fit(df[cols])
            df[cols] = mm_scaler.transform(df[cols])
            dfs.append(df)
            self.normalizers[t] = mm_scaler
            
        self.df = pd.concat(dfs)
        self.df.sort_values(by = 'end', ascending = True, inplace = True)
        
    def make(self):
        if self.config.prep_for_train:
            path = os.path.join(self.config.data_dir, 
                                'norm_pretrain_volume_bars.pqt' if self.config.objective_pretrain else  'norm_train_volume_bars.pqt')
        else:
            path = os.path.join(self.config.data_dir, 'norm_infer_volume_bars.pqt')
        
        if os.path.exists(path):
            self.df = pd.read_parquet(path)
        else:
            self.load_volume_bars()
            self.normalize()
            self.df.to_parquet(path)
            norm_filename = 'crypto_normalizer.pkl' if self.config.prep_for_train else 'crypto_infer_normalizer.pkl'
            with open(os.path.join(self.config.data_dir, norm_filename), 'wb') as f:
                pickle.dump(self.normalizers, f)
        return self.df
    
## DATASET
class MyDataset(Dataset):
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.data = self.prepare_data()
        
    def prepare_data(self):
        cols = ['open', 'high', 'low', 'close']     # 'volume', 'num_timestamps']
        momentum_targets = ['close', 'high', 'dispersion_macd', 'delta_1']
        price_change_targets = ['delta_1', 'delta_2', 'delta_3', 'delta_4', 'delta_5', 
                                'delta_6', 'delta_7', 'delta_8', 'delta_9', 'delta_10']
        data = []
        window, step_size = 64, 1
        
        for ticker in self.config.tickers:
            sub = self.df[self.df['ticker'] == ticker]
            sub.sort_values(by = 'start', ascending = True)
        
            momentum_values = sub[momentum_targets].values.tolist()
            price_change_values = sub[price_change_targets].values.tolist()
            gadf = GramianAngularField(image_size=window, method='difference', sample_range=(0, 1))
            for i in range(len(sub)-window-1, -1, -step_size):
                patch = sub[momentum_targets][i:i+window]
                if len(patch) == window:
                    gadf_image          = [gadf.transform([sub[c][i:i+window]]) for c in cols]
                    gadf_image          = np.mean(gadf_image, axis = 0)
                    moment              = momentum_values[i + window]
                    price_changes       = price_change_values[i + window]
                    data.append({
                        'ticker'                   : ticker,
                        'start'                    : str(sub['start'].values[i + window]),
                        'image'                    : gadf_image,
                        'momentum_targets'         : moment,
                        'price_change_targets'     : price_changes
                    })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx]
        ticker, start = sample['ticker'], sample['start']
        image       = (sample['image'] - 0.5) * 2
        image       = torch.tensor(image                         , dtype = torch.float32)
        moment      = torch.tensor(sample['momentum_targets'] , dtype = torch.float32)
        price_chg   = torch.tensor(sample['price_change_targets'], dtype = torch.float32)
        
        return (ticker, start, image, moment, price_chg)
    
    def make_transform(self, augment = False):
        transform = transforms.Compose([
            transforms.ToPILImage(),                            # convert tiff frame to PIL Image
            GaussianBlurTransform(radius=2),
            transforms.ToTensor(),                              # Convert frames to tensors
            
            # images are in grey scale and already normalized between 0 and 1
            # otherwise uncomment to normalize the pixels
            # transforms.Lambda(lambda x: x/255.0)    # Normalize pixel between 0, 1
            
            ## Augmentation
            #transforms.RandomHorizontalFlip(p = 0.8 if augment else 0),             # Randomly apply horizontal flip with probability 0.8
            #transforms.RandomRotation(self.max_rotation_angle if augment else 0), 
        ])
        return transform
    
class GaussianBlurTransform:
    def __init__(self, radius=2):
        self.radius = radius
    
    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(self.radius))
        return img

class LstmDataset(Dataset):
    def __init__(self, config, df):
        self.config = config
        self.n_forward = self.config.n_forward
        self.df = df
        self.data = self.prepare_data()
        
    def prepare_data(self):
        price_change_targets = ['delta_1', 'delta_2', 'delta_3', 'delta_4', 'delta_5', 
                                'delta_6', 'delta_7', 'delta_8', 'delta_9', 'delta_10']
        cols = ['open', 'high', 'low', 'close']  # 'volume', 'num_timestamps']
        momentum_targets = ['open', 'high', 'low', 'close']
        
        data = []
        for ticker in tqdm(self.df['ticker'].unique(), total = self.df['ticker'].nunique(), leave = False):
            sub = self.df[self.df['ticker'] == ticker]
            sub.sort_values(by = 'start', ascending = True, inplace = True)
            momentum_values = sub[momentum_targets].values.tolist()
            price_change_values = sub[price_change_targets].values.tolist()
            
            window, step_size = self.config.max_len, 1
            for i in range(len(sub)-window-self.n_forward, -1, -step_size):
                patch = sub[momentum_targets][i:i+window]
                if len(patch) == window:
                    inputs              = sub[cols][i:i+window].values
                    moment              = momentum_values[i + window + self.n_forward - 1]
                    price_changes       = 0 #price_change_values[i + window + self.n_forward - 1]
                    data.append({
                        'ticker'                   : ticker,
                        'start'                    : str(sub['start'].values[i + window]),
                        'end'                      : str(sub['end'].values[i + window]),
                        'inputs'                   : (inputs - 0.5) * 2,
                        'momentum_targets'         : moment,
                        'price_change_targets'     : price_changes
                        })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx]
        ticker, start, end = sample['ticker'], sample['start'], sample['end']
        #inputs       = (sample['inputs'] - 0.5) * 2
        
        image       = torch.tensor(sample['inputs']              , dtype = torch.float32).permute(1, 0)
        moment      = torch.tensor(sample['momentum_targets']    , dtype = torch.float32)
        price_chg   = 0 #torch.tensor(sample['price_change_targets'], dtype = torch.float16)
        
        #image = image.permute(1, 0)
        
        return (ticker, start, end, image, moment, price_chg)

def get_dataloaders(config, fold = 1):
    
    # data
    prep = Preprocesser(config)
    df = prep.make()
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df.sort_values(by = ['ticker', 'end'], ascending = True)

    if config.objective_pretrain:
        config.tickers = df['ticker'].unique().tolist()
    
    # fold
    folds = config.folds[fold] if not config.objective_pretrain else config.pretrain_folds[1]
    (start, end) = folds['train']
    train = df[(df['start'] >= start) & (df['end'] < end)]
    
    (start, end) = folds['valid']
    valid = df[(df['start'] >= start) & (df['end'] < end)]
    
    ## Subsample tickers
    if config.objective_pretrain:
        n_train_tics = config.pretrain_config['n_train_tickers']
        n_valid_tics = config.pretrain_config['n_valid_tickers']
        sub_train   = train[train['ticker'].isin(sorted(train['ticker'].unique().tolist())[:n_train_tics])]
        valid       = valid[valid['ticker'].isin(sorted(valid['ticker'].unique().tolist())[:n_valid_tics])]
    
    print(len(train), len(valid), valid.head(10_000)['ticker'].nunique())
    
    # dataloaders
    train_loader = DataLoader(LstmDataset(config, (sub_train if config.objective_pretrain else train) if not config.sample_run else train.head(10_000)),
                              batch_size = config.train_batch_size,
                              shuffle       = True,
                              drop_last     = False)
    
    valid_loader = DataLoader(LstmDataset(config, valid if not config.sample_run else valid.head(10_000)),
                              batch_size    = config.valid_batch_size,
                              shuffle       = False,
                              drop_last     = False)
        
    return train, train_loader, valid_loader, None
    
def get_batchloader(train, bidx, config):
    
    os.makedirs(config.batch_dir, exist_ok=True)    # make directory if not exist
   
    batch_indices   = config.batch_indices
    tickers         = sorted(train['ticker'].unique().tolist())
    start           = 0 if bidx == 0 else batch_indices[bidx - 1]
    end             = min(batch_indices[bidx], len(tickers))
    data            = train[train['ticker'].isin(tickers[start:end])]
    
    batch_path = os.path.join(config.batch_dir, f'dataset_{bidx+1}.pkl')
    if os.path.exists(batch_path):
        with open(batch_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = LstmDataset(config, data)
        # save
        with open(batch_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    train_loader = DataLoader(dataset       = dataset,
                              batch_size    = config.train_batch_size,
                              shuffle       = True,
                              drop_last     = False)
    
    del data; _ = gc.collect()
    return train_loader
            
if __name__ == '__main__':
    from helper_config import Config
    config = Config()
    config.data_dir = '/Users/Oruganti/Downloads/data'
    config.prep_for_train = False
    prep = Preprocesser(config)
    df = prep.make()