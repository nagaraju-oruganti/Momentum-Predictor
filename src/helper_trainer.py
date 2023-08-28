import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp              # mixed precision
from torch import autocast
from torchsummary import summary
from datetime import datetime

from sklearn.metrics import mean_squared_error

## Local imports
from helper_models import CNNRegressor, HSLSTMRegressor
from helper_dataset import get_dataloaders, get_batchloader

import warnings
warnings.filterwarnings('ignore')

import gc

#### Evaluate
def evaluate(model, dataloader, device):
    starts, tickers = [], []
    y_trues, y_preds = [], []
    batch_loss_list = []
    model.eval()
    with torch.no_grad():
        for (ticker, start, inputs, targets, _) in dataloader:
            
            logits, loss = model(inputs.to(device), targets.to(device))
            batch_loss_list.append(loss.item())
            
            # save
            starts.extend(list(start))
            tickers.extend(list(ticker))
            y_trues.extend(targets.to('cpu').numpy().tolist())
            y_preds.extend(logits.to('cpu').numpy().tolist())
    
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    
    loss = np.mean(batch_loss_list) ** 0.5
    rmse = mean_squared_error(y_trues, y_preds) ** 0.5
    
    eval_results = [(starts[i], tickers[i], y_trues[i, :], y_preds[i,: ]) for i in range(y_trues.shape[0])]
    
    return rmse, loss, eval_results

#### Trainer
def trainer(config, train, model, train_loader, valid_loader, optimizer, scheduler):
    
    def update_que():
        que.set_postfix({
            'batch_loss'        : f'{loss.item():4f}',
            'epoch_loss'        : f'{np.mean(batch_loss_list):4f}',
            'learning_rate'     : optimizer.param_groups[0]["lr"],
            })
    
    def save_checkpoint(model, epoch, eval_results, best = False):
        if best:
            save_path = os.path.join(config.dest_path, f'model{config.fold}.pth')
            if config.save_checkpoint:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }
                torch.save(checkpoint, save_path)
            
            # save evaluation results
            with open(os.path.join(config.dest_path, f'eval_results{config.fold}.pkl'), 'wb') as f:
                pickle.dump(eval_results, f)
                
            print(f'>>> [{datetime.now()}] - Checkpoint and predictions saved')
        
    def dis(x): return f'{x:.6f}'
        
    def run_evaluation_sequence(ref_score, counter):
        
        def print_result():
            print('')
            text =  f'>>> [{datetime.now()} | {epoch + 1}/{NUM_EPOCHS} | Early stopping counter {counter}] \n'
            text += f'    loss          - train: {dis(train_loss)}      valid: {dis(valid_loss)} \n'
            text += f'    rmse_sum      - train: {dis(train_rmse)}      valid: {dis(valid_rmse)} \n'
            text += f'    learning rate        : {optimizer.param_groups[0]["lr"]:.5e}'
            print(text + '\n')
        
        # Evaluation
        train_rmse, train_loss, _ = evaluate(model, train_loader, device) 
        valid_rmse, valid_loss, eval_results = evaluate(model, valid_loader, device)
        
        # append results
        lr =  optimizer.param_groups[0]["lr"]
        results.append((epoch, train_loss, valid_loss, train_rmse, valid_rmse, lr, bidx))
        
        # Learning rate scheduler
        eval_metric = valid_loss
        scheduler.step(eval_metric)           # apply scheduler on validation accuracy
        
        ### Save checkpoint
        if ((epoch + 1) > config.save_epoch_wait):
            save_checkpoint(model, epoch, eval_results, best = eval_metric < ref_score)
        
        # Tracking early stop
        counter = 0 if eval_metric <= ref_score else counter + 1
        ref_score = min(ref_score, eval_metric)
        done = counter >= config.early_stop_count
        
        # show results
        print_result()
        
        # Save results
        with open(os.path.join(config.dest_path, f'results{config.fold}.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        return ref_score, counter, done 
    
    ### MIXED PRECISION
    scaler = amp.GradScaler()
    
    results = []
    device = config.device
    precision = torch.bfloat16 if str(device) == 'cpu' else torch.float16
    NUM_EPOCHS = config.num_epochs
    iters_to_accumlate = config.iters_to_accumlate
    
    # dummy value for placeholders
    eval_results = []
    ref_score, counter = 1e3, 0
    train_loss, valid_loss, train_f1, valid_f1 = 0, 0, 0, 0
    
    ## Evaluation baseline before training
    print('Baseline:')
    epoch, bidx = -1, 0
    ref_score, counter, done = run_evaluation_sequence(ref_score, counter)
    
    for epoch in range(NUM_EPOCHS):
        for bidx in range(config.train_batches):
            
            ## Process next batch of training dataset, if multiple batches are used
            if config.train_batches > 1:
                del train_loader; _ = gc.collect()
                train_loader = get_batchloader(train, bidx, config)
                
            model.train()
            batch_loss_list = []
            que = tqdm(enumerate(train_loader), total = len(train_loader), 
                       desc = f'Epoch {epoch + 1}' + f" | batch {(bidx+1)}/{config.train_batches}" if config.train_batches > 1 else "")
            for i, (_, _, images, targets, _) in que:
                
                ###### TRAINING SECQUENCE            
                with autocast(device_type = str(device), dtype = precision):
                    _, loss = model(images.to(device), targets.to(device))            # Forward pass
                    if config.train_on_rmse:
                        loss **= 0.5
                    loss = loss / iters_to_accumlate
                
                # - Accmulates scaled gradients    
                scaler.scale(loss).backward()           # scale loss
                
                if (i + 1) % iters_to_accumlate == 0:
                    scaler.step(optimizer)                  # step
                    scaler.update()
                    optimizer.zero_grad()
                #######
                
                batch_loss_list.append(loss.item())
                
                # Update que status
                update_que()
                
            ### Run evaluation sequence
            ref_score, counter, done = run_evaluation_sequence(ref_score, counter)
            if done:
                return results
            
    return results

def save_config(config, path):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

def train(config):
    print();print('-'*50);print(); print(f'Training fold {config.fold}')
    
    device = config.device
    
    config.dest_path = os.path.join(config.models_dir, config.model_name)
    os.makedirs(config.dest_path, exist_ok=True)
    
    # define model
    #model = CNNRegressor(config, num_outputs=10)
    model = HSLSTMRegressor(input_size  = config.max_len, 
                            hidden_size = config.max_len * 2,
                            num_layers  = config.num_lstm_layers, 
                            output_size = 4, 
                            device      = device,
                            dropout_prob = config.dropout_prob,
                            fine_tune   = config.fine_tune)
    
    if config.pretrain_config['load_weights_from'] != None:
        path = os.path.join(config.models_dir, config.pretrain_config['load_weights_from'])
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
        print(f'Loaded pretrained weights from path: {path}')
        
    model.to(device)
    
    # optmizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, patience=4)
    
    # dataloaders
    train, train_loader, valid_loader, _ = get_dataloaders(config, fold = config.fold)
    
    # Trainer
    results = trainer(config, train, model, train_loader, valid_loader, optimizer, scheduler)
    
    ### SAVE RESULTS
    with open(os.path.join(config.dest_path, f'results{config.fold}.pkl'), 'wb') as f:
        pickle.dump(results, f)
        
    return results


if __name__ == '__main__':
    from helper_config import Config
    
    config = Config()
    config.fold = 1
    config.data_dir = '/Users/Oruganti/Downloads/data'
    config.models_dir = 'models' 
    config.model_name = 'hslstm_delete'
    config.iters_to_accumlate = 1
    config.sample_run = True
    config.learning_rate = 1e-3
    config.num_epochs = 200
    config.save_epoch_wait = 1    
    config.early_stop_count = 20
    config.save_checkpoint = True
    config.time_steps = 4
    config.n_forward = 2
    
    results = train(config)
