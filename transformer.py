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
import numpy as np
import torch
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from SAnD.utils.inference import Inference_SoH_Siamese, Inference_SoH_Normal, Inference_SoH_Normal_Improve
from SAnD.utils.functions import save_example_to_csv
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
######################################## Nueva rama normal_improve 12/03/2025


# Real Dataset Generator
#dataFile = 'dataset/ARC-FY/B0025'   # Modify this path
#raw = scio.loadmat(dataFile)['B0025'][0][0][0][0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
cycles.pop()
assert (len(cycles) == len(labels)), 'Number of measurements not matched!'

print(f"cantidad de ciclos: {len(cycles)}")
print(f"cantidad de labels: {len(labels)}")

data = []

# Filtrar solo los ciclos y etiquetas que no estén vacíos
filtered_cycles = []
filtered_labels = []

for i in range(len(labels)):
    if len(labels[i]) > 0:  # Solo conservar si la etiqueta no está vacía
        filtered_cycles.append(cycles[i])
        filtered_labels.append(labels[i])
for i in range(len(labels)):
    if labels[i] > 0.5:  # Solo conservar si la etiqueta es mayor de 0.5
        filtered_cycles.append(cycles[i])
        filtered_labels.append(labels[i])

# Sustituimos las listas originales por las filtradas
cycles = filtered_cycles
labels = filtered_labels

print(f"Nueva cantidad de ciclos: {len(cycles)}")
print(f"Nueva cantidad de labels: {len(labels)}")
for lb in range(len(labels)):
    labels[lb] = labels[lb][0] / 1.856487420818157  # TODO: first (largest) capacity found, but probably not the full cp
# calculate SOHs
# for lb in range(len(labels)):
#     #print(f"label {lb} de un total de {len(labels)}")
#     if (1974 < lb < 1979) or (2006 < lb < 2027):
#         labels[lb] = labels[lb+20][0] / 1.856487420818157  # TODO: first (largest) capacity found, but probably not the full cp
#     else:
#         lb =2106
#         labels[lb] = labels[lb][0] / 1.856487420818157  # TODO: first (largest) capacity found, but probably not the full cp
#         #print(f"⚠️ Error en lb={lb}: labels[{lb}] está vacío.")
# # print(f"Len data: {len(data)}, Len labels: {len(labels)}")  # Comprobar si siguen coincidiendo
labels = labels * 20
# labels = labels * 1
# # print(f"Len data: {len(data)}, Len labels: {len(labels)}")  # Comprobar si siguen coincidiendo
#
for t0 in [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15, 16.5, 18, 19.5, 21, 22.5, 24, 25.5, 27, 28.5]:
# for t0 in [0]:
    for cy in cycles:
        t0 = 0
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
    #         # print(f"En iteración {t0}, tamaño actual de data: {len(data)}")

print(f"Final Len data: {len(data)}, Len labels: {len(labels)}")

# Data shape: (495, 401, 3)
# Labels shape: (495)
# data = cycles
for i in range(len(data)):
    mm = MinMaxScaler()
    data[i] = mm.fit_transform(data[i])
data = np.array(data)
data = data[: , 0:400 ,:]
print(data.shape)
data=torch.from_numpy(data).type(torch.FloatTensor)
labels=torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

# Escalar etiquetas
label_scaler = MinMaxScaler()
# labels = label_scaler.fit_transform(np.array(labels).reshape(-1, 1)).flatten()
labels = label_scaler.fit_transform(np.asarray(labels).reshape(-1, 1)).flatten()


# Dividir en train, val y test (estratificado si `labels` tiene clases desbalanceadas)
x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Convertir a tensores
# x_train = torch.tensor(x_train, dtype=torch.float32)
# x_val = torch.tensor(x_val, dtype=torch.float32)
# x_test = torch.tensor(x_test, dtype=torch.float32)

x_train = x_train.clone().detach().float()
x_val = x_val.clone().detach().float()
x_test = x_test.clone().detach().float()

# y_train = y_train.clone().detach().float()
# y_val = y_val.clone().detach().float()
# y_test = y_test.clone().detach().float()


y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Shuffle los datos (opcional si `train_test_split` ya los aleatoriza)
indices = torch.randperm(len(x_train))
x_train, y_train = x_train[indices], y_train[indices]

# Crear DataLoaders
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)


# plt.hist(y_train, bins=20, edgecolor='black', alpha=0.7)
# plt.xlabel("SoH")
# plt.ylabel("Frecuencia")
# plt.title("Distribución de SoH en el dataset de entrenamiento")
# plt.show()
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

#################################################################################################################3
# Training
###############################################################################################################
in_feature = 3
seq_len = 400
n_heads = 1
factor = 1
num_class = 1
num_layers = 8

hyperparameters = {
    "in_feature": in_feature,
    "seq_len": seq_len,
    "n_heads": n_heads,
    "factor": factor,
    "num_class": num_class,
    "num_layers": num_layers
}


clf = NeuralNetworkClassifier(
    SiameseSAnD(SAnD_Embedding(in_feature, seq_len, n_heads, factor, num_class, num_layers)),
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    SAnDImprove(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    ContrastiveLoss(),
    nn.MSELoss(),
    nn.MSELoss(),
    #nn.SmoothL1Loss(beta=0.1),  # Cambiar a SmoothL1Loss,
    # optim.AdamW,optimizer_config={"lr": 1e-6, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    optim.AdamW,optimizer_config={"lr": 1e-6, "betas": (0.9, 0.96), "eps": 1e-08, "weight_decay": 1e-6},
    # optim.SGD, optimizer_config={"lr":1e-6, "momentum": 0.9,"weight_decay": 1e-4},
    #experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
    experiment = Experiment(api_key="Td3ICbNoK8hW14nwxZfp10SGN",
                            project_name="nn4soh",
                            workspace="javiergranadocirce")


)
inference = False
if inference == True:
    train = False
elif inference == False:
    train = True

export_csv = False
if export_csv == True:
    inference = False
    train = False
    save_example_to_csv(x_test, y_test, 2490, filename="save_params/ciclo_de_carga.csv")

if train ==  True:
    # # training network Normal
    # clf.fit_normal(x_train, y_train, x_val, y_val, x_test, y_test,
    #         {"train": train_loader,
    #       "val": val_loader,
    #       "test": test_loader},
    #       epochs=80
    # )

    # #training network Improve
    clf.fit_normal_improve(x_train, y_train, x_val, y_val, x_test, y_test,
             {"train": train_loader,
          "val": val_loader,
          "test": test_loader},
          epochs=80
    )
    # # # #
    # training network Siamese
    # clf.fit_siamese(x_train, y_train, x_val, y_val, x_test, y_test,
    #             {"train": train_loader,
    #         "val": val_loader,
    #         "test": test_loader},
    #         epochs=80
    # )
    # # # #
    # # # #
    # # # #
    # # # # #Inference SoH Siames ###############################
    # # # # #inference_model = Inference_SoH_Siamese("save_params/trained_model_siamese.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
    # # # # #soh_predictions = inference_model.predict(test_loader)
    # # # # #Inference SoH ###############################
    # # # #
    # # # # #Inference SoH Normal ###############################
    # # # # # inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
    # # # # # soh_predictions = inference_model.predict(test_loader)
    # # # # #Inference SoH ###############################
    # # # #
    # # # # # evaluating
    # # # # # clf.restore_from_file("save_params/trained model.pth", "cuda")
    # # # # # clf.evaluate(test_loader)
    # # # #
    # # # # save
    # clf.save_to_file_normal("save_params/")
    clf.save_to_file_normal_improve("save_params/")
    #clf.save_to_file_siamese("save_params/")
    # #
    # # #
    # # #
    # # #
    # # # # Conversión a ONNX
    # # # # Cargar el modelo entrenado
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super(WrappedModel, self).__init__()
            self.model = model
            self.sigmoid = nn.Sigmoid()  # Agregar sigmoide

        def forward(self, x):
            return self.sigmoid(self.model(x))  # Aplicar sigmoide después del modelo

    # modelo = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)
    modelo = SAnDImprove(in_feature, seq_len, n_heads, factor, num_class, num_layers)
    # modelo = SAnD_Embedding(in_feature, seq_len, n_heads, factor, num_class, num_layers)
    # # # # Verificar los atributos de modelo_siamese
    # print(modelo_siamese)
    # # # # # Verificar los atributos de modelo_normal
    # print(modelo)
    # # # # # Copiar pesos de la parte compartida del modelo siamesa al modelo normal
    # # # # Transferir pesos del modelo siamesa al modelo normal
    # modelo.encoder.load_state_dict(modelo_siamese.sand.encoder.state_dict(), strict=False)  # Transferir encoder
    # modelo.dense_interpolation.load_state_dict(modelo_siamese.sand.dense_interpolation.state_dict(), strict=False)  # Transferir dense_interpolation
    # # #
    # # #
    # # # print("Pesos transferidos correctamente.")
    # # # print(modelo)
    # # #
    # # # # 2. Cargar el diccionario de estado correctamente

    # checkpoint = torch.load("save_params/trained_model_normal_old.pth", map_location="cpu")
    # print(checkpoint.keys())  # Ver qué hay dentro
    # if "hyperparameters" in checkpoint:  # Si guardaste los hiperparámetros
    #     print(checkpoint["hyperparameters"])
    checkpoint = torch.load("save_params/trained_model_normal_improve.pth", map_location="cpu")
    # checkpoint = torch.load("save_params/trained_model_normal.pth", map_location="cpu")
    modelo.load_state_dict(checkpoint["model_state_dict"], strict=False)
    modelo.eval()
    wrapped_model = WrappedModel(modelo)  # Envolver modelo con sigmoide
    # #
    # # # # Crear un dummy input (ajusta el tamaño según tu entrada real)
    # # #
    input_shape = (400, 3)
    dummy_input = torch.randn(1, *input_shape)
    # # #
    # # # # Exportar a ONNX
    # torch.onnx.export(modelo, dummy_input, "save_params/trained_model_normal.onnx", opset_version=13)
    torch.onnx.export(wrapped_model, dummy_input, "save_params/trained_model_normal_improve.onnx", opset_version=13)
    # torch.onnx.export(modelo, dummy_input, "save_params/trained_model_normal_improve_old.onnx", opset_version=13)
    # #
    # # # #
    #
    # #
# ##################################################################################################
# # #Inferencia en PC
# #############################################################################################
#
def cargar_modelo(modo="onnx", modelo = None):
    """Carga el modelo según el modo especificado."""
    if modo == "onnx":
        # session = ort.InferenceSession("save_params/trained_model_normal.onnx")
        session = ort.InferenceSession(modelo)
        input_name = session.get_inputs()[0].name
        return session, input_name
    # elif modo == "pth":
    #     #inference_model = Inference_SoH_Normal_Improve("save_params/trained_model_normal_improve.pth", input_features=3, seq_len=400, n_heads=64, factor=32, n_class=1, n_layers=12)
    #     inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
    #     return inference_model
    else:
        raise ValueError("Modo no reconocido. Usa 'onnx' o 'pth'.")

def realizar_inferencia(x_test, y_test, test_loader, modo="onnx", modelo=None):
    """Realiza la inferencia usando ONNX o PyTorch y calcula métricas."""


    if modo == "onnx":
        predicciones = []
        etiquetas_reales = []
        mae_total, mse_sum, mape_total = 0, 0, 0
        modelo, input_name = cargar_modelo(modo, modelo)
        for idx in range(len(x_test)):
            x_sample = x_test[idx].numpy().astype(np.float32)  # Convertir tensor a numpy
            x_sample = np.expand_dims(x_sample, axis=0)  # Añadir batch dimension

            # Inferencia con ONNX
            output = modelo.run(None, {input_name: x_sample})[0]
            # # Inferencia con PyTorch
            # else:
            #     with torch.no_grad():
            #         x_tensor = torch.tensor(x_sample)
            #         output = inference_model.predict(x_tensor)

            # Guardar predicción y etiqueta real
            pred = output[0]  # Asumimos salida en la primera posición
            real = y_test[idx].item()
            predicciones.append(pred)
            etiquetas_reales.append(real)

            # Cálculo de errores
            mae_total += np.abs(pred - real)
            mse_sum += (pred - real) ** 2
            if real != 0:
                mape_total += np.abs((pred - real) / real)

            # Mostrar resultado parcial
            print(f"Ejemplo {idx + 1}/{len(x_test)} -> Predicción: {pred}, Etiqueta Real: {real}")

    if modo == "pth":
        predicciones = []
        etiquetas_reales = []
        mae_total, mse_sum, mape_total, smap_total = 0, 0, 0, 0
        # sand_model = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)
        # # Cargar los pesos del modelo entrenado
        # checkpoint = torch.load("save_params/trained_model_normal.pth", map_location=device)
        # sand_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # sand_model.to(device)
        # sand_model.eval()
        #
        # with torch.no_grad():
        #     for idx in range(len(x_test)):
        #         x_sample = x_test[idx].clone().detach().to(device)
        #         soh_raw = sand_model(x_sample)  # Obtener SoH
        #         pred = torch.sigmoid(soh_raw).cpu().numpy()
        #         real = y_test[idx].item()
        #         predicciones.append(pred)
        #         etiquetas_reales.append(real)

        #Inference SoH Normal ###############################
        # inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
        inference_model = Inference_SoH_Normal_Improve(modelo, input_features=3, seq_len=400, n_heads=1, factor=1, n_class=1, n_layers=8)
        # inference_model = Inference_SoH_Siamese(modelo, input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
        soh_predictions = inference_model.predict(test_loader)
        #Inference SoH ###############################

        # real = soh_predictions[1]
        real_values = []
        pred_values = []

        for idx in range(len(x_test)):
            pred = soh_predictions[0][idx]
            real = soh_predictions[1][idx]

            # Guardar valores para graficar
            real_values.append(real)
            pred_values.append(pred)

            # Cálculo de errores
            mae_total += np.abs(real - pred)
            mse_sum += (real - pred) ** 2
            if real != 0:
                mape_total += np.abs((pred - real) / real)
            smap_sup = pred - real
            smap_inf = (np.abs(pred) + np.abs(real)) / 2
            smap_total += np.abs(smap_sup / smap_inf)

            # Mostrar resultado parcial
            print(f"Ejemplo {idx + 1}/{len(x_test)} -> Predicción: {pred}, Etiqueta Real: {real}")

        # Graficar los valores reales y predichos
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(real_values[:100])), real_values[:100], label="Real", color="blue", marker="o")
        plt.scatter(range(len(pred_values[:100])), pred_values[:100], label="Predicho", color="red", marker="x")


# Etiquetas y título
        plt.xlabel("Índice de muestra")
        plt.ylabel("State of Health (SoH)")
        plt.title("Comparación de SoH Real vs Predicho")
        plt.legend()
        plt.show()
    # Cálculo de métricas
    mae = mae_total / len(x_test)
    mse = mse_sum / len(x_test)
    rmse = np.sqrt(mse)
    smape = smap_total / len(x_test)
    # Calcula el MAPE promedio
    mape = mape_total / len(x_test)
    #  Multiplica por 100 para tener el resultado en porcentaje
    # mape_total*= 100
    #mape = (mape_total / len(x_test)) * 100

    print("\nMétricas finales:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"SMAPE: {smape}")

# 🔹 Ejemplo de uso
if inference ==  True:
    modo = "pth"  # Cambia a "pth" para usar el modelo original
    modelo ="save_params/trained_model_normal_improve.pth"
    realizar_inferencia(x_test, y_test, test_loader, modo, modelo)


