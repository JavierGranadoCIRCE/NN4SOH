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
from SAnD.utils.inference import Inference_SoH_Siamese, Inference_SoH_Normal
import scipy.io as scio
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import numpy as np

from SAnD.core.modules import ContrastiveLoss

from SAnD.core.modules import ContrastiveLoss

from SAnD.core.model import SAnD, SAnD_Embedding, SiameseSAnD, SAnDImprove
from SAnD.utils.functions import generar_pares_aleatorios
from SAnD.utils.trainer import NeuralNetworkClassifier

######################################## Nueva rama normal_improve 12/03/2025


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

#dataFile = 'dataset/ARC-FY/B0005'   # Modify this path
#raw = scio.loadmat(dataFile)['B0005'][0][0][0][0]

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
    #print(f"label {lb} de un total de {len(labels)}")
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
x_train = data[:7023]
x_val = data[7023: 7093]
x_test = data[7093:]
y_train = labels[:7023] ##7023
y_val = labels[7023: 7093]#7023: 7093
y_test = labels[7093:]#7093
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_ds, batch_size=16)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)


# plt.hist(y_train, bins=20, edgecolor='black', alpha=0.7)
# plt.xlabel("SoH")
# plt.ylabel("Frecuencia")
# plt.title("Distribución de SoH en el dataset de entrenamiento")
# plt.show()

#x1_cont, x2_cont, y_cont = generar_pares_aleatorios(x_train, y_train, umbral_soh=0.02)

# ##########################################################################
# # PLoteo de los coclos de carga del dataset completo
#
#
# # Etiquetas de las variables
# variables = ["Tensión (V)", "Corriente (A)", "Temperatura (°C)"]
# colores = ["b", "r", "g"]  # Azul, rojo y verde
#
# for i in range(x_train.shape[0]):  # Recorremos los ciclos de carga
#     plt.figure(figsize=(10, 5))
#
#     # Dibujar las 3 variables en distintos colores
#     for j in range(3):
#         plt.plot(x_train[i, :, j], color=colores[j], label=variables[j])
#
#     soh_value = y_train[i]  # Obtener el SoH del ciclo actual
#     plt.xlabel("Tiempo (puntos de muestreo)")
#     plt.ylabel("Valor")
#     plt.title(f"Ciclo de carga {i+1} - SoH: {soh_value:.2f}%")  # Agregar el SoH en el título
#     plt.legend()
#     plt.grid()
#
#
#     plt.show()
#
#     input("Presiona Enter para ver el siguiente ciclo...")  # Espera antes de mostrar el siguiente gráfico
#     plt.close()
# # PLoteo de los coclos de carga del dataset completo
# ##########################################################################



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
n_heads = 64
factor = 32
num_class = 1
num_layers = 12



clf = NeuralNetworkClassifier(
    SiameseSAnD(SAnD_Embedding(in_feature, seq_len, n_heads, factor, num_class, num_layers)),
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    SAnDImprove(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    ContrastiveLoss(),
    nn.MSELoss(),
    nn.MSELoss(),
    #nn.SmoothL1Loss(),  # Cambiar a SmoothL1Loss,
    #optim.AdamW,optimizer_config={"lr": 1e-4, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    optim.AdamW,optimizer_config={"lr": 1e-3, "betas": (0.9, 0.98), "eps": 1e-09, "weight_decay": 5e-4},
    #experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
    experiment = Experiment(api_key="Td3ICbNoK8hW14nwxZfp10SGN",
                            project_name="nn4soh",
                            workspace="javiergranadocirce")


)


# training network Normal
# clf.fit_normal(x_train, y_train, x_val, y_val, x_test, y_test,
#          {"train": train_loader,
#       "val": val_loader,
#       "test": test_loader},
#       epochs=80
#  )

# training network Improve
clf.fit_normal_improve(x_train, y_train, x_val, y_val, x_test, y_test,
         {"train": train_loader,
      "val": val_loader,
      "test": test_loader},
      epochs=80
 )

# training network Siamese
# clf.fit_siamese(x_train, y_train, x_val, y_val, x_test, y_test,
#             {"train": train_loader,
#         "val": val_loader,
#         "test": test_loader},
#         epochs=80
# )



#Inference SoH Siames ###############################
#inference_model = Inference_SoH_Siamese("save_params/trained_model_siamese.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
#soh_predictions = inference_model.predict(test_loader)
#Inference SoH ###############################

#Inference SoH Normal ###############################
# inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
# soh_predictions = inference_model.predict(test_loader)
#Inference SoH ###############################

# evaluating
# clf.restore_from_file("save_params/trained model.pth", "cuda")
# clf.evaluate(test_loader)

# save
#clf.save_to_file_normal("save_params/")
clf.save_to_file_normal_improve("save_params/")
#clf.save_to_file_siamese("save_params/")




# # Conversión a ONNX
# # Cargar el modelo entrenado
# modelo = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)
modelo = SAnDImprove(in_feature, seq_len, n_heads, factor, num_class, num_layers)
# modelo_siamese = SiameseSAnD(SAnD_Embedding(in_feature, seq_len, n_heads, factor, num_class, num_layers))
# # Verificar los atributos de modelo_siamese
# # print(modelo_siamese)
# # # Verificar los atributos de modelo_normal
# # print(modelo_normal)
# # # Copiar pesos de la parte compartida del modelo siamesa al modelo normal
# # Transferir pesos del modelo siamesa al modelo normal
# modelo.encoder.load_state_dict(modelo_siamese.sand.encoder.state_dict())  # Transferir encoder
# modelo.dense_interpolation.load_state_dict(modelo_siamese.sand.dense_interpolation.state_dict())  # Transferir dense_interpolation
#
#
# print("Pesos transferidos correctamente.")
# print(modelo)
#
# # 2. Cargar el diccionario de estado correctamente
# checkpoint = torch.load("save_params/trained_model_normal.pth", map_location="cpu")
checkpoint = torch.load("save_params/trained_model_normal_improve.pth", map_location="cpu")
# # checkpoint = torch.load("save_params/trained_model_siamese.pth", map_location="cpu")
modelo.load_state_dict(checkpoint["model_state_dict"])  # Extrae solo "model_state_dict"
# modelo.eval()
#
# # Crear un dummy input (ajusta el tamaño según tu entrada real)
#
input_shape = (400, 3)
dummy_input = torch.randn(1, *input_shape)
#
# # Exportar a ONNX
# # torch.onnx.export(modelo, dummy_input, "save_params/trained_model_normal.onnx", opset_version=13)
# torch.onnx.export(modelo, dummy_input, "save_params/trained_model_siamese.onnx", opset_version=13)
torch.onnx.export(modelo, dummy_input, "save_params/trained_model_normal_improve.onnx", opset_version=13)

#
#
# #Inferencia en PC
# Cargar el modelo ONNX
session = ort.InferenceSession("save_params/trained_model_normal_improve.onnx")
# session = ort.InferenceSession("save_params/trained_model_siamese.onnx")

# Obtener nombres de las entradas del modelo
input_name = session.get_inputs()[0].name
print(f"Nombre de la entrada: {input_name}")

# # Crear un dummy input (ajustar si es necesario)
# dummy_input = np.random.randn(1, 400, 3).astype(np.float32)
#
# # Realizar la inferencia
# outputs = session.run(None, {input_name: dummy_input})

# Inicializar las listas para las predicciones y etiquetas reales
predicciones = []
etiquetas_reales = []

# Inicializar las variables para los cálculos de error
mae_total = 0
mse_total = 0
mse_sum = 0
mape_total = 0

# Recorrer todos los ejemplos de test
for idx in range(len(x_test)):
    # Seleccionar un ejemplo de test
    x_sample = x_test[idx].numpy().astype(np.float32)  # Convertir de tensor a numpy
    x_sample = np.expand_dims(x_sample, axis=0)  # Añadir batch dimension

    # Realizar la inferencia
    outputs = session.run(None, {input_name: x_sample})

    # Guardar la predicción y la etiqueta real
    predicciones.append(outputs)
    etiquetas_reales.append(y_test[idx].item())

    # Calcular los errores para este ejemplo
    pred = outputs[0]  # Asumiendo que la salida de la inferencia está en la primera posición
    real = y_test[idx].item()

    # Calcular el error absoluto
    mae_total += np.abs(pred - real)

    # Calcular el error cuadrático
    mse_sum += (pred - real) ** 2

    # Calcular la desviación relativa media (MAPE)
    if real != 0:  # Evitar división por cero
        mape_total += np.abs((pred - real) / real)

    # Mostrar resultado parcial
    print(f"Ejemplo {idx + 1}/{len(x_test)} -> Predicción: {pred}, Etiqueta Real: {real}")

# Cálculo de las métricas finales
mae = mae_total / len(x_test)
mse = mse_sum / len(x_test)
rmse = np.sqrt(mse)
mape = (mape_total / len(x_test)) * 100  # Convertir a porcentaje

# Mostrar los resultados finales
print(f"\nMétricas finales:")
print(f"MAE (Error Absoluto Medio): {mae}")
print(f"MSE (Error Cuadrático Medio): {mse}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse}")
print(f"MAPE (Desviación Relativa Media): {mape}%")

# # # Obtener la etiqueta real
# # y_real = y_test[idx].item()  # Convertir a valor escalar si es necesario
# #
# # # Mostrar la salida
# # print("Resultado de la inferencia:", outputs)
# # print(f"Etiqueta real: {y_real}")
