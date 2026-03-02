import sys
import random
sys.setrecursionlimit(15000)
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import torch.utils.data as Data
import os
import matplotlib
from sklearn import metrics
matplotlib.use('Agg')
from data_processing import load_data,generate_perturbed_data
from MyModel import iDNA_SMP
from sklearn.model_selection import KFold
np.random.seed(1377)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 50
Margin = 2
SAVE_DIR = ''
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')
filename = ''
[X_train, y_train, X_valid, y_valid, X_test, y_test] = load_data(filename)
X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = torch.cat((y_train, y_valid), axis=0)
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
exclude_regions = [(16, 17), (20, 22), (25), (27, 28)]
X_unlabeled = generate_perturbed_data(
    X_train,
    perturb_ratio=0.1,
    exclude_regions=exclude_regions
)
import torch.utils.data as Data
train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)
batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TripleLoss(torch.nn.Module):
    def __init__(self, margin=Margin):
        super(TripleLoss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)
def collate(batch):
    anchors = []
    positives = []
    negatives = []
    label_ls = []
    batch_size = len(batch)
    labels = [item[1] for item in batch]
    for i in range(batch_size):
        anchor_seq, anchor_label = batch[i][0], batch[i][1]
        positive_indices = [j for j in range(batch_size) if labels[j] == anchor_label and j != i]
        if not positive_indices:
            positive_idx = i
        else:
            positive_idx = random.choice(positive_indices)
        negative_indices = [j for j in range(batch_size) if labels[j] != anchor_label]
        if not negative_indices:
            negative_idx = i
        else:
            negative_idx = random.choice(negative_indices)
        anchors.append(anchor_seq.unsqueeze(0))
        positives.append(batch[positive_idx][0].unsqueeze(0))
        negatives.append(batch[negative_idx][0].unsqueeze(0))
        label_ls.append(anchor_label.unsqueeze(0))
    if len(anchors) < batch_size:
        raise ValueError(f"Generated {len(anchors)} samples, but expected {batch_size}.")
    anchors = torch.cat(anchors).to(device)
    positives = torch.cat(positives).to(device)
    negatives = torch.cat(negatives).to(device)
    label = torch.cat(label_ls).to(device)
    return anchors, positives, negatives, label
    
train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,drop_last=True)

model = Semisup6mA()
kf = KFold(n_splits=5)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

def zhibiao(predict_proba, predict_class, Y_test_array):
    predict_proba = predict_proba
    Y_test_array = Y_test_array.reshape([len(Y_test_array),-1]).tolist()
    predict_class = predict_class.reshape([len(predict_class),-1]).tolist()
    acc = accuracy_score(Y_test_array, predict_class)
    binary_acc = metrics.accuracy_score(Y_test_array, predict_class)
    precision = metrics.precision_score(Y_test_array, predict_class)
    recall = metrics.recall_score(Y_test_array, predict_class)
    f1 = metrics.f1_score(Y_test_array, predict_class)
    auc = metrics.roc_auc_score(Y_test_array, predict_proba)
    AP = average_precision_score(Y_test_array, predict_proba)
    mcc = metrics.matthews_corrcoef(Y_test_array, predict_class)
    TN, FP, FN, TP = metrics.confusion_matrix(Y_test_array, predict_class).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)
    return acc, auc,mcc,sensitivity,specificity,AP
    
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
    
net = Semisup6mA().to(device)
lr = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
criterion = TripleLoss()
criterion_model = nn.CrossEntropyLoss(reduction='mean')
best_auc = 0
checkpoint = torch.load('Fusionmodel/F.vesca.pl', map_location=device)
net.load_state_dict(checkpoint['model'])
data_array = np.loadtxt("z_score_data-F.vesca.csv", delimiter=",", dtype=np.float32)
z_score_tensor = torch.from_numpy(data_array)

def get_confidence_threshold(auc, base=0.5, max_increase=0.1):
    import numpy as np
    increase = max_increase * np.sqrt(auc)
    threshold = base + increase
    return np.clip(threshold, 0.5, 0.7)
    
def generate_pseudo_labels(model_path, unlabeled_data, device, threshold=0.5):
    model = Semisup6mA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        outputs = model.trainModel(torch.tensor(unlabeled_data).to(device),z_score_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        pseudo_labels = (probabilities >= threshold).int().cpu()
    return pseudo_labels, probabilities.cpu()
    
num_iterations = 50
dynamic_threshold = 0.5

for iteration in range(num_iterations):
    print(f"\n=== Iteration {iteration + 1} ===")
    pseudo_labels, probabilities = generate_pseudo_labels(
        model_path='',
        unlabeled_data=X_unlabeled,
        device=device,
        threshold=0.5
    )
    X_combined = torch.cat([X_train, torch.tensor(X_unlabeled)], dim=0)
    y_combined = torch.cat([y_train, pseudo_labels], dim=0)
    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    combined_dataset = CombinedDataset(X_combined, y_combined)
    combined_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    net.train()
    for anchors, positives, negatives, labels in combined_loader:
        anchors, positives, negatives, labels = anchors.to(device), positives.to(device), negatives.to(
            device), labels.to(device)
        output_anchor = net(anchors,z_score_tensor.to(device))
        output_positive = net(positives,z_score_tensor.to(device))
        output_negative = net(negatives,z_score_tensor.to(device))
        loss_triplet = criterion(output_anchor, output_positive, output_negative)
        output_class = net.trainModel(anchors,z_score_tensor.to(device))
        loss_class = criterion_model(output_class, labels)
        loss = loss_triplet + loss_class
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        outputs = net.trainModel(X_test.to(device),z_score_tensor.to(device))
        outputs = outputs.cpu().detach().numpy()
        predict = np.vstack(outputs[:, 1])
        threshold = 0.5
        y_pred = (predict >= threshold).astype(int)
        acc, auc, mcc, sensitivity, specificity,AP = zhibiao(predict, y_pred, y_test)
        auc = metrics.roc_auc_score(y_test, predict)
    # dynamic_threshold = get_confidence_threshold(auc, base=0.5, max_increase=0.1)


