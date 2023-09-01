import os
import pandas as pd
import numpy as np
import pickle

##### PORTFOLIO SIMULATION
## Date range from 01-01-2020 to 12-31-2022
## Intelligence from pretrained model
## Testing on the 10-crytpo assets

## Since the dataset is transformed into Volume bars, the only value align with price bars is 
## close price and we will use it when making trading decisions

## Rules:
## - Initial investment 1 million
## - Cost of transacation is fixed at 0.05% for buy and 0.03% for sell (this includes slippage as well)
## - No shorts are allowed
## - 10% of initial portfolio value is equally split between the assets
## - Cash not invested will be saved in the respective asset bucket
## - At the end of the simulation the portfolios are cashed-out
## - Capital gain tax on the profits is 35% and the investment manager sells the stocks to cover at the end of each year (Dec 30).

## Scenerios:
## - Baseline: weights are not adjusted 
##     + passive asset manager who invest and leave the composition unchanged for 2 years
## - Simulation (agent1): weights are adjusted as per predictions
##     + No more than 5% of cash will be invested on a transaction
##     + Sell value is in proportion to the change in price. Large drop in price could result in large sale of shares.
##          + # of shares for sale = (%p x # value of assets) / (per share value at the time of prediction)
## Metrics:
## - Although there is not feedback to the simulator the two metrics will be monitored

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from helper_models import CNNRegressor, HSLSTMRegressor, TransformerModel
from helper_dataset import get_dataloaders

from sklearn.metrics import mean_squared_error

class PortfolioSimulator:
    def __init__(self, config):
        self.config     = config
        self.dest_dir   = config.models_dir + '/' + config.model_name
        self.sim_config = config.sim_config
        os.makedirs(self.dest_dir, exist_ok = True)
        self.reset()
        
    def reset(self):
        self.initial_investment = self.sim_config['initial_balance']
        self.buy_txn_cost       = self.sim_config['buy_txn_cost']
        self.sell_txn_cost      = self.sim_config['sell_txn_cost']
        self.history            = {t: [] for t in self.config.tickers}
        
    def prep_model(self):
        self.model = HSLSTMRegressor(input_size  = self.config.max_len, 
                                     hidden_size = self.config.max_len * 2,
                                     num_layers  = self.config.num_layers, 
                                     output_size = 4, 
                                     device      = self.config.device,
                                     dropout_prob = self.config.dropout_prob,
                                     fine_tune   = self.config.fine_tune)

        path = os.path.join(config.models_dir, self.sim_config['load_weights_from'])
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
        
        print(f'Loaded pretrained weights from path: {path}')
        self.model.to(self.config.device)
        
    def make_predictions(self):
        for n_forward in [1, 2, 3, 4, 5, 6, 7]:
            print(); print('-'*100); print(f'making predictions for lookahead timestep: {n_forward}')
            self.config.n_forward = n_forward
            _, _, valid_loader, _ = get_dataloaders(self.config, fold = 1000)
            
            rmse, loss, eval_results = self.evaluate(model = self.model, 
                                                    dataloader = valid_loader, 
                                                    device = self.config.device)
            
            print(f'RMSE: {rmse:.6f}')
            with open(f'{self.dest_dir}/eval_results_t{n_forward}.pkl', 'wb') as f:
                pickle.dump(eval_results, f)
            
    def evaluate(self, model, dataloader, device):
        starts, ends, tickers = [], [], []
        y_trues, y_preds = [], []
        batch_loss_list = []
        model.eval()
        with torch.no_grad():
            for (ticker, start, end, inputs, targets, _) in dataloader:
                
                logits, loss = model(inputs.to(device), targets.to(device))
                batch_loss_list.append(loss.item())
                
                # save
                starts.extend(list(start))
                ends.extend(list(end))
                tickers.extend(list(ticker))
                y_trues.extend(targets.to('cpu').numpy().tolist())
                y_preds.extend(logits.to('cpu').numpy().tolist())
        
        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)
        
        loss = np.mean(batch_loss_list) ** 0.5
        rmse = mean_squared_error(y_trues, y_preds) ** 0.5
        
        eval_results = [(starts[i], ends[i], tickers[i], y_trues[i, :], y_preds[i,: ]) for i in range(y_trues.shape[0])]
        
        return rmse, loss, eval_results
    
if __name__ == '__main__':
    from helper_config import Config
    config = Config()
    
    config.prep_for_train = False
    config.data_dir = 'data'
    config.models_dir = 'models'
    config.model_name = 'simualtion'
    config.sim_config['load_weights_from'] = '/Users/Oruganti/Downloads/data/model1.pth'
    config.max_len = 256
    config.num_layers = 4
    
    simulator = PortfolioSimulator(config = config)
    simulator.reset()
    simulator.prep_model()
    simulator.make_predictions()