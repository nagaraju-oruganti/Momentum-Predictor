# GAFIMAGE CREATOR
class GAFCreator:
    def __init__(self, config):
        self.params = config.params
        self.features = config.features
        self.tickers = config.tickers
        
    def make_feature_images(self, subset):
        '''
            For every subset of timeseries data, 
            we are creating image where each channel is a feature
        '''
        
        gasf = GramianAngularField(**self.params)
        images = [gasf.transform(subset[c].values.reshape(1, -1)) for c in subset[self.features]]
        images = np.vstack(images)
        return images
    
    def make_feature_for_window(self, start, end):
        subset = self.df[(self.df['Datetime'] >= start) & (self.df['Datetime'] <= end)]
        tick_images = [self.make_feature_images(subset[subset['tic'] == tic]) for tic in self.tickers]
        tick_images = np.hstack(tick_images).astype(np.float16)
        return tick_images
            
    def make_data(self, df, dest_path, kind = 'train'):
        self.df = df
        dates = list(sorted(self.df['Datetime'].unique()))
        n = len(dates)
        w = self.params['image_size'] 
        batch_num = 0
        data = {}
        for i in tqdm(range(n-w-1, -1, -1), total = n-1):
            start, end = dates[i], dates[i+w-1]
            data[i] = {
                '3dimage' : self.make_feature_for_window(start, end),
                'labels'  : [self.df[(self.df['Datetime'] == dates[i+w]) & (self.df['tic'] == tic)]['label'].values[0] for tic in self.tickers]
            }
        
            ## save batches
            if len(data) == 500:
                self.save(data, dest_path, name = f'{kind}_{batch_num}')
                data = {}
                batch_num += 1
            
        ## final save
        self.save(data, dest_path, name = f'{kind}_{batch_num}')
    
    ## SAVE BATCHES
    def save(self, data, dest_path, name):
        print(f'Saved {name}, {len(data)}')
        with open(os.path.join(dest_path, f'{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)

### FEATURING ENGINEERING
class FeatureEngineering:
    def __init__(self, config):
        self.config = config
        #self.remove = self.config.remove
        self.remove_tics = ['1000SHIB', 'ICP', 'LDO']
        
        # Features should be considered dynamically
        self.features = ['Open', 'High', 'Low', 'Volume', 'Close']
        
    def load_data(self):
        # load data
        self.df = pd.read_csv(f'{self.config.data_dir}/long_TICKERS.csv')
        
        # convert Datetime to pandas DateTime object
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        
        # Make year and quarter features so that the normalization happens every quarter
        self.df['year'] = self.df['Datetime'].dt.year
        self.df['quarter'] = self.df['Datetime'].dt.quarter
        
        # TICKERS
        self.tickers = self.df['tic'].unique().tolist()
        self.tickers = [t for t in self.tickers if t not in self.remove_tics]
                
    def normalize_by_tic(self, subset):
        dfs = []
        for tic in self.tickers:
            sub = subset[subset['tic'] == tic]
            if len(sub) == 0:continue
            for f in self.features:
                min_val = min(sub[f])
                max_vl = max(sub[f])
                sub[f] = (sub[f] - min_val) / (max_vl - min_val)
            dfs.append(sub)
        
        # join
        normalized_subset = pd.concat(dfs)
        normalized_subset.sort_values(by = 'Datetime', ascending = True, inplace = True)
        return normalized_subset
        
    def normalize(self):
        dfs = []
        for y in self.df['year'].unique():
            for q in self.df['quarter'].unique():
                subset = self.df[(self.df['year'] == y) & (self.df['quarter'] == q)]
                
                # Make sure I have the timestamps aligned
                timestamps = list(set(subset.groupby('Datetime').filter(lambda x: len(x) >= len(self.tickers))['Datetime']))
                
                subset = subset[subset['Datetime'].isin(timestamps)]
                if len(subset) == 0: continue
                norm_subset = self.normalize_by_tic(subset)
                
                dfs.append(norm_subset)
        self.df = pd.concat(dfs)
        self.df.sort_values(by = 'Datetime', ascending = True, inplace = True)

    def make_gasf_images(self, start, end, dest_path, kind):
        self.config.tickers = self.tickers
        self.config.features = self.features
        gasf = GAFCreator(config = self.config)
        df = self.df[(self.df['Datetime']>= start) & (self.df['Datetime']<= end)]
        gasf.make_data(df, dest_path, kind)
        
    def make(self):
        
        self.load_data()
        self.normalize()
        
        ## MAKE IMAGES
        for kind in ['train', 'valid', 'test']:
            self.dest_dir = os.path.join(self.config.data_dir, kind)
            os.makedirs(self.dest_dir, exist_ok=True)
            start, end = self.config.splits[kind]
            self.make_gasf_images(start, end, self.dest_dir, kind = kind)
        