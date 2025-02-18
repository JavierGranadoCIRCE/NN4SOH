from comet_ml import Experiment
import os
import time
import tqdm
import pandas as pd
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io as scio
import glob
import os


from SAnD.core.modules import ContrastiveLoss

from SAnD.core.modules import ContrastiveLoss

from SAnD.core.model import SAnD, SAnD_Embedding, SiameseSAnD
from SAnD.utils.trainer import NeuralNetworkClassifier
######################################## Commit nueva rama merge entre Dataset complet y entrenamiento Siames 18022025

# Real Dataset Generator
#dataFile = 'dataset/ARC-FY/B0025'   # Modify this path
#raw = scio.loadmat(dataFile)['B0025'][0][0][0][0]

data_folder = "dataset/ARC-FY/"  # Modifica esto según tu estructura de carpetas
mat_files = glob.glob(os.path.join(data_folder, "*.mat"))
# Lista para almacenar los datos concatenados
raw = []
# Cargar cada archivo y agregar sus datos a la lista `raw`
for mat_file in mat_files:
    data = scio.loadmat(mat_file)
    key = list(data.keys())[-1]  # Toma la última clave que suele ser el nombre del dataset
    extracted_data = data[key][0][0][0][0]  # Extrae los datos
    raw.extend(extracted_data)  # Concatenar los datos a la lista

print(f"Se han cargado {len(mat_files)} archivos. Tamaño total de raw: {len(raw)}")

# dataFile = 'dataset/ARC-FY/B0005'   # Modify this path
# raw = scio.loadmat(dataFile)['B0005'][0][0][0][0]

# raw data parsing
cycles = []
labels = []
for i in range(len(raw)):
    if raw[i][0] == ['charge']:
        if i+1 != len(raw) and raw[i+1][0] != ['charge'] and len(raw[i][3][0][0][0][0]) > 850: # discard unfair records
            cycles.append(raw[i][3][0][0])
            if raw[i+1][0] == ['discharge']:
                labels.append(raw[i+1][3][0][0][6][0])
            elif i+2 != len(raw) and raw[i+2][0] == ['discharge']:
                labels.append(raw[i+2][3][0][0][6][0])
assert (len(cycles) == len(labels)), 'Number of measurements not matched!'

data = []
# calculate SOHs
for lb in range(len(labels)):
    print(f"label {lb} de un total de {len(labels)}")
    if (1974 < lb < 1979) or (2006 < lb < 2027):
        labels[lb] = labels[lb+20][0] / 1.856487420818157  # TODO: first (largest) capacity found, but probably not the full cp
    else:
        labels[lb] = labels[lb][0] / 1.856487420818157  # TODO: first (largest) capacity found, but probably not the full cp
labels = labels * 3

for t0 in [0, 1.5, 3]:
    for cy in cycles:
        t = t0
        t_limit = 4000 + t0  # TODO: this parameter can be further tuned
        cursor = 0
        cy_new = []
        while cursor <= len(cy[0][0]) and t <= t_limit:
            while cy[5][0][cursor] <= t:
                cursor += 1
            x1 = cy[5][0][cursor - 1]
            x2 = cy[5][0][cursor]
            point = []
            for i in range(3):
                y1 = cy[i][0][cursor - 1]
                y2 = cy[i][0][cursor]
                y = (t - x1) * (y2 - y1) / (x2 - x1) + y1
                point.append(y)
            cy_new.append(point)
            cursor -= 1
            t += 10
        data.append(cy_new)


# Data shape: (495, 401, 3)
# Labels shape: (495)
for i in range(len(data)):
    mm = MinMaxScaler()
    data[i] = mm.fit_transform(data[i])
data = np.array(data)
data = data[: , 0:400 ,:]
print(data.shape)
data=torch.from_numpy(data).type(torch.FloatTensor)
labels=torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

# data_set = list(zip(data, labels))
# np.random.shuffle(data_set)
# data, labels = data_set[0], data_set[1]
ax_train = data[:7023]
x_val = data[7023: 7093]
x_test = data[7093:]
y_train = labels[:7023]
y_val = labels[7023: 7093]
y_test = labels[7093:]
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_ds, batch_size=1)
val_loader = DataLoader(val_ds, batch_size=1)
test_loader = DataLoader(test_ds, batch_size=1)

# Fake Dataset Generater
# x_train = torch.randn(1024, 256, 23)    # [N, seq_len, features]
# x_val = torch.randn(128, 256, 23)       # [N, seq_len, features]
# x_test =  torch.randn(512, 256, 23)     # [N, seq_len, features]

# y_train = torch.randint(0, 9, (1024, ))
# y_val = torch.randint(0, 9, (128, ))
# y_test = torch.randint(0, 9, (512, ))


# train_ds = TensorDataset(x_train, y_train)
# val_ds = TensorDataset(x_val, y_val)
# test_ds = TensorDataset(x_test, y_test)

# train_loader = DataLoader(train_ds, batch_size=128)
# val_loader = DataLoader(val_ds, batch_size=128)
# test_loader = DataLoader(test_ds, batch_size=128)

# Training
in_feature = 3
seq_len = 400
n_heads = 32
factor = 32
num_class = 1
num_layers = 4

clf = NeuralNetworkClassifier(
    SiameseSAnD(SAnD_Embedding(in_feature, seq_len, n_heads, factor, num_class, num_layers)),
    ContrastiveLoss(),
    optim.Adam, optimizer_config={"lr": 1e-4, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    #experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
    experiment = Experiment(api_key="Td3ICbNoK8hW14nwxZfp10SGN",
                        project_name="nn4soh",
                        workspace="javiergranadocirce")


)

# from SAnD.core.model import SAnD
# from SAnD.core.modules import RegressionModule


# class RegSAnD(SAnD):
#     def __init__(self, *args, **kwargs):
#         super(RegSAnD, self).__init__(*args, **kwargs)
#         d_model = kwargs.get("d_model")
#         factor = kwargs.get("factor")
#         output_size = kwargs.get("n_class")    # output_size

#         self.clf = RegressionModule(d_model, factor, output_size)


# # model = RegSAnD(
# #     input_features=..., seq_len=..., n_heads=..., factor=...,
# #     n_class=..., n_layers=...
# # )

# clf = NeuralNetworkClassifier(
#     RegSAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
#     nn.CrossEntropyLoss(),
#     optim.Adam, optimizer_config={"lr": 1e-5, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
#     experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
# )

# training network
clf.fit(
    {"train": train_loader,
     "val": val_loader,
     "test": test_loader},
    #epochs=80
     epochs=80
)

# evaluating
# clf.evaluate(test_loader)

# save
clf.save_to_file("save_params/")

#Conversión a TorchScript para ejecutar el modelo en Raspberry o ARM sin depender de PyTorch en tiempo real
# Cargar el modelo
model = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)  # Define tu modelo
model.load_state_dict(torch.load("save_params/trained model.pth"), strict=False)
model.eval()

# Convertir a TorchScript
#scripted_model = torch.jit.script(model)

# Guardar el modelo convertido
#scripted_model.save("save_params/modelo_torchscript.pt")
