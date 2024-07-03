import os
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import Dropout

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' AlphaPose 17 body keypoints
0 nose
1 left eye
2 right eye
3 left ear
4 right ear
5 left shoulder
6 right shoulder
7 left elbow
8 right elbow 
9 left wrist
10 right wrist
11 left hip
12 right hip
13 left knee
14 right knee
15 left ankle
16 right ankle
'''

skeleton_edges_alphapose = [
    # Head to Shoulders
    (0, 5), (0, 6),   # Nose to Shoulders
    (1, 2),           # Eyes
    (2, 4), (1, 3),   # Eyes to Ears
 
    # Arms
    (5, 7), (7, 9),   # Left arm
    (6, 8), (8, 10),  # Right arm

    # Torso
    (5, 6),           # Shoulders
    (5, 11), (6, 12), # Shoulders to Hips
    (11, 12),         # Hips
    
    # Legs
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16)  # Right leg
]

def keypoints_to_graph(flattened_keypoints, edge_index):
    return Data(x=flattened_keypoints, edge_index=edge_index)
graph_batches = []

class TemporalDynamicGCN(nn.Module):
    def __init__(self, window_size, num_frames, num_persons, num_keypoints, num_features, num_classes, hidden_dim=128, \
                 lstm_hidden=256, num_lstm_layers=1, dropout_rate=0.5, model_path=None):
        super(TemporalDynamicGCN, self).__init__()
        self.window_size = window_size
        self.num_frames = num_frames
        self.num_persons = num_persons
        self.num_keypoints = num_keypoints
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        #self.dropout = Dropout(p=0.5)
        
        # GCN Conv layers
        self.conv1 = pyg_nn.GCNConv(num_features, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, num_lstm_layers, batch_first=True, dropout=dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden, num_classes)

        if model_path is not None and os.path.exists(model_path):
            try:
                self.load_state_dict(torch.load(model_path, map_location=device))
                self.to(device)
                print("Model loaded successfully from:", model_path)
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")

    def forward(self, all_keypoint_batches):

        graph_batches = []
        lstm_input = []
        batch_graphs = []

        for keypoint_batch in all_keypoint_batches:
            batch_graphs = []           
            # Convert each tensor in the keypoint_batch to a graph object
            for keypoint_tensor in keypoint_batch:
                batch_keypoints_tensor = torch.from_numpy(np.array(keypoint_tensor)).float()
                edge_index = torch.tensor([[start_node, end_node] for start_node, end_node in skeleton_edges_alphapose], dtype=torch.long).t().contiguous()
                flattened_tensor = batch_keypoints_tensor.view(-1, 2)
                graph_data = keypoints_to_graph(flattened_tensor, edge_index)
                batch_graphs.append(graph_data)
            # Append the list of graph objects for the current batch to graph_batches
            graph_batches.append(batch_graphs)
        
        # move the data to gpu        
        graph_batches_gpu = []

        for batch_graphs in graph_batches:
            batch_graphs_gpu = []
            for data_batch in batch_graphs:
                data_batch_gpu = data_batch.to(device)  
                batch_graphs_gpu.append(data_batch_gpu)
            graph_batches_gpu.append(batch_graphs_gpu)

        #print ('length of graph_batches_gpu', len(graph_batches_gpu), len(graph_batches_gpu[0]), graph_batches_gpu[0][0])

        # use gcn         
        for batch_graphs in graph_batches_gpu:
            pooled_batch = []
            for data_batch in batch_graphs:
                x, edge_index = data_batch.x, data_batch.edge_index
                x = self.conv1(x, edge_index)
                x = nn.functional.relu(x)
                x = self.conv2(x, edge_index)
                x = nn.functional.relu(x)
                pooled = global_mean_pool(x, batch=None)  # Compute global mean pooling per graph
                pooled_batch.append(pooled)
            lstm_input.append(pooled_batch)

        #print ('lstm_input lenth', len(lstm_input), len(lstm_input[0]), lstm_input[0][0].shape)
        window_size = len(all_keypoint_batches[0])
        #print ('window_size ', window_size)

        # Initialize an empty list to store the concatenated tensors
        concatenated_tensors = []

        # Iterate over each batch in lstm_input
        for batch in lstm_input:
            # Concatenate the tensors along the batch dimension
            concatenated_batch = torch.cat(batch, dim=0)
            # Reshape the concatenated batch to [1, num_windows, features]
            reshaped_batch = concatenated_batch.view(1, -1, concatenated_batch.size(-1))
            # Append the reshaped batch to the list
            concatenated_tensors.append(reshaped_batch)

        lstm_input_tensor = torch.cat(concatenated_tensors, dim=0)
        #print ('shape of lstm_input_tensor', lstm_input_tensor.shape)

        # LSTM layer
        lstm_output, _ = self.lstm(lstm_input_tensor)
        #print ('lstm_output shape', lstm_output.shape)
        # use max pooling instead of mean to capture the most significant features in each dimension across time. 
        
        #lstm_output = lstm_output.mean(dim=1)
        lstm_output = torch.max(lstm_output, dim=1)[0]  
        #print(f"LSTM OUT2 : {lstm_output}")
        
        # Calculate mean and max pooling 내가 추가함.
        # lstm_output_mean = lstm_output.mean(dim=1)
        # lstm_output_max = torch.max(lstm_output, dim=1)[0]  
        
        # Fully connected layer for mean 내가 추가함.
        # out_mean = self.fc(lstm_output_mean)
        # score_mean = torch.sigmoid(out_mean).squeeze(1)
        
        # Fully connected layer for max 내가 추가함
        # out_max = self.fc(lstm_output_max)
        # score_max = torch.sigmoid(out_max).squeeze(1)
        
        # Fully connected layer 원본 코드
        out = self.fc(lstm_output)
        #print ('out', out)
        #print(f"LSTM OUT : {out}")
        score = torch.sigmoid(out).squeeze(1)
        #print ('score', score)
        #print(f"LSTM OUT : {score}")
        return score
   
    
    def load_model(self, model_path):
        if model_path is not None:
            self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.to(device)
            self.load_state_dict(torch.load(self.model_path, map_location=device))
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

        return  
    

def evaluate_frames(model, all_keypoint_batches):
    #print ('all_keypoint_batches', len(all_keypoint_batches), len(all_keypoint_batches[0]), len(all_keypoint_batches[0][0]))
    model.eval()
    with torch.no_grad():  # Turn off gradients for prediction
        all_scores = model.forward(all_keypoint_batches)
        # print(f"Sigmoid OUT : {all_scores}")
        # 텐서의 길이 구하기
        length = all_scores.size(0)
        
        
        # option 1 take max score
        max_score = round(torch.max(all_scores).item(),3)
        mean_score = round(torch.mean(all_scores).item(),3)
        
        # option 2 Mean of Top Values After Removing Lowest
        
        # sorted_scores = all_scores.sort(descending=True).values
        # #print(f"sorted_scores OUT : {sorted_scores}")
        # middle_scores = sorted_scores[:-5]
        # #print(f"middle_scores OUT : {middle_scores}")
        # mean_score = middle_scores.mean().item()
        # print(f"mean_score OUT : {mean_score}")
        

        # option3  Mean of Scores Above a Threshold
        '''
        filtered_scores = all_scores[all_scores > 0.01]
        #print ('filtered_scores---------------------------------', filtered_scores)
        if filtered_scores.numel() > 0:  # Checking if there are any elements left after filtering
            score = filtered_scores.mean().item()
        else:
            score = 0
        '''

    return max_score, mean_score

def evaluate_frames_one_input(model, all_keypoint_batches): # 최대값만
    #print ('all_keypoint_batches', len(all_keypoint_batches), len(all_keypoint_batches[0]), len(all_keypoint_batches[0][0]))
    model.eval()

    with torch.no_grad():  # Turn off gradients for prediction
        all_scores = model.forward(all_keypoint_batches)
                
        # option 1 take max score
        max_score = torch.max(all_scores).item()

        # option 2 Mean of Top Values After Removing Lowest
        
        sorted_scores = all_scores.sort(descending=True).values
        middle_scores = sorted_scores[:-5]
        mean_score = middle_scores.mean().item()
        

        # option3  Mean of Scores Above a Threshold
        '''
        filtered_scores = all_scores[all_scores > 0.01]
        #print ('filtered_scores---------------------------------', filtered_scores)
        if filtered_scores.numel() > 0:  # Checking if there are any elements left after filtering
            score = filtered_scores.mean().item()
        else:
            score = 0
        '''

    return max_score

def evaluate_frames_with_kf(model, all_keypoint_batches, kf):
    model.eval()

    with torch.no_grad():  # Turn off gradients for prediction
        # Obtain scores from the model for all batches
        scores = model(all_keypoint_batches)
        # Convert scores to CPU and numpy, reshape for Kalman filter
        # Reshape to (-1, 1) if kf.update expects a 2D column vector per timestamp
        scores_np = scores.cpu().numpy().reshape(-1, 1)
        
        # Apply Kalman filter to each score individually and store results
        filtered_scores = np.array([kf.update(score) for score in scores_np])
        
        # Find the maximum score from the filtered scores
        max_score = np.max(filtered_scores)
        
    return max_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class KalmanFilter:
    def __init__(self, F=np.array([[1]]), H=np.array([[1]]), Q=np.array([[0.1]]), R=np.array([[1.0]]), P=np.array([[1.0]]), x_est=np.array([[0]])):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x_est = x_est

    def update(self, z):
        # Convert z to a 2D array if it's not
        if z.ndim == 1:
            z = np.array([z])

        # Time Update (Predict)
        x_pred = self.F @ self.x_est
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement Update (Correct)
        K = P_pred @ self.H.T / (self.H @ P_pred @ self.H.T + self.R)
        self.x_est = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(len(self.P)) - K @ self.H) @ P_pred

        return self.x_est.squeeze()

