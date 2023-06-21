import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def take_norm(data, cellwise_norm=True, log1p=True):
    data_norm = data.copy()
    data_norm = data_norm.astype('float32')
    if cellwise_norm:
        libs = data.sum(axis=1)
        norm_factor = np.diag(np.median(libs) / libs)
        data_norm = np.dot(norm_factor, data_norm)

    if log1p:
        data_norm = np.log2(data_norm + 1.)
    return data_norm
    
def find_hv_genes(X, top=1000):
    ngene = X.shape[1]
    CV = []
    for i in range(ngene):
        x = X[:, i]
        x = x[x != 0]
        mu = np.mean(x)
        var = np.var(x)
        CV.append(var / mu)
    CV = np.array(CV)
    rank = CV.argsort()
    hv_genes = np.arange(len(CV))[rank[:-1 * top - 1:-1]]
    return hv_genes

def pytorch_categorical_ce(y, logit, reduce_mean=True):
    cce = -torch.sum(torch.log(F.softmax(logit, dim=-1)) * F.one_hot(y.type(torch.int64), logit.shape[1]), axis=-1)
    if reduce_mean:
        cce = torch.mean(cce)
    return cce

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def cluster(input_data, npc, n_clusters, ncell, ngene):
    if n_clusters > 1:
        n = np.min((ncell, ngene))
        pca = PCA(n_components=n)
        pcs = pca.fit_transform(input_data)
        var = (pca.explained_variance_ratio_).cumsum()
        npc_raw = (np.where(var > 0.7))[0].min()  # number of PC used in K-means
        if npc_raw > npc:
            npc_raw = npc
        pcs = pcs[:, :npc_raw]
        # K-means clustering on PCs
        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(StandardScaler().fit_transform(pcs))
        clustering_label = kmeans.labels_
        return to_categorical(y=clustering_label, num_classes=len(np.unique(clustering_label)))
    else:
        return None

class AutoClass(nn.Module):
    def __init__(self, ncell, ngene, encoder_layer_size, dropout_rate, n_clusters):
        super(AutoClass, self).__init__()
        self.ncell = ncell
        self.ngene = ngene
        self.n_clusters = n_clusters
        self.dropout_rate = dropout_rate
        self.encoder_layer_size = encoder_layer_size
        self.len_layer = len(self.encoder_layer_size)
        
        encoder_layer_size = self.encoder_layer_size.copy()
        encoder_layer_size.insert(0, self.ngene)
        len_layer = len(encoder_layer_size)
        
        #encoder
        self.encoders = nn.Sequential()
        for i in range(len_layer-1):
            self.encoders.add_module("encoder_layer" + str(i+1), nn.Linear(encoder_layer_size[i], encoder_layer_size[i+1]))
            self.encoders.add_module("encoder_activation" + str(i+1), nn.ReLU())
        
        #bottleneck
        self.bottleneck_dropout = nn.Dropout(self.dropout_rate)
        
        #decoder
        self.decoders = nn.Sequential()
        for i in reversed(range(len_layer-1)):
            self.decoders.add_module("decoder_layer" + str(i+1), nn.Linear(encoder_layer_size[i+1], encoder_layer_size[i]))
            if i == 0:
                self.decoders.add_module("output_activation", nn.Softplus())
            else:
                self.decoders.add_module("decoder_activation" + str(i+1), nn.ReLU())
        
        #classifier
        self.classifier_layer = nn.Linear(encoder_layer_size[-1], self.n_clusters)
        self.classifier_activation = nn.Softmax(dim=-1)
        
        #weight initalization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)
    
    def forward(self, x):
        #input and encoder layer
        x = self.encoders(x)
        
        #apply bottleneck dropout
        x = self.bottleneck_dropout(x)
        
        #decoder and output layer
        x_imp = self.decoders(x)
        
        #classifier layer
        x_classified =self.classifier_activation(self.classifier_layer(x))
        
        return x_imp, x_classified
    
    def _set_freeze_(self, status):
        for n,p in self.features.named_parameters():
            p.requires_grad = status
        # for m in self.features.children():
        #     for p in m.parameters():
        #         p.requires_grad=status    


    def freeze_feature_layers(self):
        self._set_freeze_(False)

    def unfreeze_feature_layers(self):
        self._set_freeze_(True)
        
def training(AC, data, dummy_label, lr, batch_size, epochs, classifier_weight, reg, reduce_lr, early_stopping, verbose):
    #load criterion, optimizer, scheduler
    criterion_decoder = nn.MSELoss().to(device)
    criterion_classifier = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(AC.parameters(), lr=lr, weight_decay=reg)
    #optimizer.add_param_group({"params": AC.classifier_layer.parameters(), "lr": lr})
    #optimizer.add_param_group({"params": model.fc_gender.parameters(), "lr": 0.1})
    if reduce_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=reduce_lr, verbose=verbose)
    if early_stopping:
        early_stopper = EarlyStopper(patience=early_stopping, min_delta=0)

    #load data
    train_data = []
    for i in range(len(data)):
        train_data.append([data[i], dummy_label[i]])

    trainloader = torch.utils.data.DataLoader(train_data[:int(len(train_data)*0.9)], shuffle=True, batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(train_data[int(len(train_data)*0.9):], shuffle=True, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=batch_size)
    
    #train,val the model
    for epoch in range(epochs):
        AC.to(device)

        #train
        train_loss = 0
        with torch.set_grad_enabled(True):
            AC.train()
            for inp_data, labels in trainloader:
                optimizer.zero_grad()
                out_imp, out_classified = AC(inp_data.to(device))
                loss_imp = criterion_decoder(out_imp, inp_data.to(device))
                #loss_clusters = criterion_classifier(out_classified.to(torch.float), labels.to(device).to(torch.float))
                #loss_clusters = nn.NLLLoss().to(device)(torch.log(out_classified), labels.to(device).argmax(-1))
                loss_clusters = pytorch_categorical_ce(labels.to(device).argmax(-1), out_classified)
                #loss_clusters = nn.CrossEntropyLoss()(out_classified.float(), labels.to(device).argmax(dim=-1))
                loss = (1 - classifier_weight) * loss_imp + classifier_weight * loss_clusters
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        train_loss = train_loss / len(trainloader)

        #val
        val_loss = 0
        with torch.no_grad():
            AC.eval()
            for inp_data, labels in valloader:
                optimizer.zero_grad()
                out_imp, out_classified = AC(inp_data.to(device))
                loss_imp = criterion_decoder(out_imp, inp_data.to(device))
                loss_clusters = pytorch_categorical_ce(labels.to(device).argmax(-1), out_classified)
                loss = (1 - classifier_weight) * loss_imp + classifier_weight * loss_clusters
                val_loss += loss.item()
        val_loss = val_loss / len(valloader)
        if reduce_lr:
            scheduler.step(val_loss)
        if early_stopping:
            if early_stopper.early_stop(val_loss):             
                break

    #predict
    imp = []
    with torch.no_grad():
        for inp_data, labels in loader:
            AC.eval()
            optimizer.zero_grad()
            out_imp, out_classified = AC(inp_data.to(device))

            imp.extend(out_imp.cpu().numpy())
    imp = np.array(imp)
    
    return imp


def autoClassImpute(data, encoder_layer_size=[128], num_clusters=9, dropout_rate=0.1, classifier_weight=0.9, truelabel=[], cellwise_norm=True, log1p=True, reg=0.000, batch_size=32, epochs=300, verbose=False, npc=15, early_stopping=30, reduce_lr=15, lr=0.001):
    
    t1 = time.time()
    data = data.astype("float32")
    data = take_norm(data, cellwise_norm=cellwise_norm, log1p=log1p)
    ncell = data.shape[0]
    ngene = data.shape[1]
    print('{} cells and {} genes'.format(ncell, ngene))
    
    imps = np.zeros((ncell, ngene))
    if type(num_clusters) == int:
        num_clusters = [np.max((1,num_clusters-1)),num_clusters,num_clusters+1]
    print('number of clusters in pre-clustering:{}'.format(num_clusters))
    for n_cluster in num_clusters:
        print('n_cluster = {}...'.format(n_cluster))
        dummy_label = cluster(data, npc, n_cluster, ncell, ngene)
        AC = AutoClass(ncell, ngene, encoder_layer_size, dropout_rate, n_cluster)
        imp = training(AC, data, dummy_label, lr, batch_size, epochs, classifier_weight, reg, reduce_lr, early_stopping, verbose)
        imps = imps + imp
    imps = imps / len(num_clusters)
    print('escape time is: {}'.format(time.time() - t1))
    return imps