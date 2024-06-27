import os
import sys
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
import glob
import natsort
import re
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# 2023년 11월 20일 추가, 상대 경로
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Actionsrecognition.Models import *
from Visualizer import plot_graphs, plot_confusion_metrix

device = 'cuda'

epochs = 50
batch_size = 256

class_names = ['Normal', 'Fall Down']

num_class = len(class_names)

def load_dataset(data_files, batch_size, split_size=0):
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return train_loader, valid_loader

def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()

def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model

if __name__ == '__main__':
    eval_skel = []
    mname_list = []
    save_folder = 'C:/Users/twoimo/Documents/GitHub/FallDetection-original/Actionsrecognition/saved2-1/'      
    skel_folder = 'C:/Users/twoimo/Documents/GitHub/FallDetection-original/Actionsrecognition/_NewPrisonClip60_skel/'
    
    for mname in os.listdir(save_folder):
        if(mname[0] == '_'):
            continue
        mname_list.append(mname)
    
    for s in os.listdir(skel_folder):
        skel_path = os.path.join(skel_folder, s)
        eval_skel.append(skel_path)
    
    mname_list = natsort.natsorted(mname_list) ##### 순서대로 리스트 정렬
    
    # MODEL.
    #for k in range(10, 110, 10):
    for i, mname in enumerate(mname_list):
        working_folder = os.path.join(save_folder, mname)

        graph_args = {'strategy': 'spatial'}
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = Adadelta(model.parameters())

        losser = torch.nn.BCELoss()

        model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)
        model.load_state_dict(torch.load(os.path.join(working_folder, 'tsstg-model.pth')))
        #model.load_state_dict(torch.load(os.path.join(working_folder, 'Fall300_Normal200_EP'+ str(k) +'_BS256_LS8.pth')))

        # EVALUATION.
        model = set_training(model, False)
        
        for j in range(len(eval_skel)):
            data_file = eval_skel[j]
            eval_loader, _ = load_dataset([data_file], batch_size)

            print('Evaluation.')
            run_loss = 0.0
            run_accu = 0.0
            y_preds = []
            y_trues = []
            with tqdm(eval_loader, desc='eval') as iterator:
                for pts, lbs in iterator:
                    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
                    mot = mot.to(device)
                    pts = pts.to(device)
                    lbs = lbs.to(device)

                    out = model((pts, mot))
                    loss = losser(out, lbs)

                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),
                                            lbs.detach().cpu().numpy())
                    run_accu += accu

                    y_preds.extend(out.argmax(1).detach().cpu().numpy())
                    y_trues.extend(lbs.argmax(1).cpu().numpy())

                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                        loss.item(), accu))
                    iterator.update()

            run_loss = run_loss / len(iterator)
            run_accu = run_accu / len(iterator)
            
            y_trues_f = np.array(y_trues)
            y_preds_f = np.array(y_preds)
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues_f, y_preds_f, average=None, zero_division=0)
            f1_score_avg = np.mean(f1_score)

            # Confusion metrix (frames)
            plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu: {:.4f}, F1-Score: {:.4f}'.format(
                os.path.basename(data_file), run_loss, run_accu, f1_score_avg
            ), None, save=os.path.join(working_folder, '{}-confusion_matrix_frames_m.png'.format(os.path.basename(data_file).split('.')[0])))
            
            # Confusion metrix (rates)
            # plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu: {:.4f}'.format(
            #     os.path.basename(data_file), run_loss, run_accu
            # ), 'true', save=os.path.join(working_folder, 'R_{}-confusion_matrix_rates_m.png'.format(os.path.basename(data_file).split('.')[0])))

            print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))