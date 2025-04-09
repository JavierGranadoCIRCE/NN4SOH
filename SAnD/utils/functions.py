import torch
import numpy as np
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict

def positional_encoding(n_positions: int, hidden_dim: int) -> torch.Tensor:
    def calc_angles(pos, i):
        rates = 1 / np.power(10000, (2*(i // 2)) / np.float32(hidden_dim))
        return pos * rates

    rads = calc_angles(np.arange(n_positions)[:, np.newaxis], np.arange(hidden_dim)[np.newaxis, :])

    rads[:, 0::2] = np.sin(rads[:, 0::2])
    rads[:, 1::2] = np.cos(rads[:, 1::2])

    pos_enc = rads[np.newaxis, ...]
    pos_enc = torch.tensor(pos_enc, dtype=torch.float32, requires_grad=False)
    return pos_enc


def dense_interpolation(batch_size: int, seq_len: int, factor: int) -> torch.Tensor:
    W = np.zeros((factor, seq_len), dtype=np.float32)
    for t in range(seq_len):
        s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
        for m in range(factor):
            tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
            w = np.power(tmp, 2, dtype=np.float32)
            W[m, t] = w

    W = torch.tensor(W, requires_grad=False).float().unsqueeze(0)
    return W.repeat(batch_size, 1, 1)


def subsequent_mask(size: int) -> torch.Tensor:
    """
    from Harvard NLP
    The Annotated Transformer

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking

    :param size: int
    :return: torch.Tensor
    """
    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("float32")
    mask = torch.from_numpy(mask) == 0
    return mask.float()

def generar_pares_aleatorios(x_train, y_train, umbral_soh=0.02):
    """
    Genera pares aleatorios de ejemplos de carga junto con sus etiquetas 0 o 1
    según si tienen un estado de salud similar o diferente.

    Parámetros:
    - data: lista de ciclos de carga (cada elemento es una secuencia de carga)
    - labels: lista de valores de SoH correspondientes a cada ciclo
    - num_pares: número total de pares a generar
    - umbral_soh: diferencia máxima entre SoH para considerar que es el mismo estado

    Retorna:
    - X_pairs: lista con los pares de ciclos de carga
    - y_pairs: lista con etiquetas 0 o 1 según su estado de salud
    """

    X_pairs = []
    y_pairs = []

    total_ciclos = len(x_train)


    # Seleccionar dos ciclos aleatorios
    i, j = random.sample(range(total_ciclos), 2)

    ciclo_1 = x_train[i]
    ciclo_2 = x_train[j]
    soh_1 = y_train[i]
    soh_2 = y_train[j]

    # Asignar etiqueta: 1 si los SoH son similares, 0 si son diferentes
    y = 1 if abs(soh_1 - soh_2) < umbral_soh else 0

    # Guardar el par y su etiqueta
    X_pairs.append((ciclo_1, ciclo_2))
    y_pairs.append(y)
    x1 = ciclo_1
    x2 = ciclo_2
    y_cont = y
    if x1.dim() == 2:  # Si x1 tiene la forma (400, 3)
        x1 = x1.unsqueeze(0)  # Convierte en (1, 400, 3)
    if x2.dim() == 2:  # Si x2 tiene la forma (400, 3)
        x2 = x2.unsqueeze(0)  # Convierte en (1, 400, 3)
    return x1, x2, y_cont


def create_cycle_triplets(data, labels):
    x_pairs = []
    capacities = []
    y_targets = []

    for i in range(1, len(data) - 1):
        x_pair = torch.stack([data[i], data[i+1]])  # (2, 400, 3)
        x_pairs.append(x_pair)
        capacities.append(labels[i])       # SoH del ciclo anterior
        y_targets.append(labels[i+1])      # SoH del ciclo actual (target)

    x_pairs = torch.stack(x_pairs)
    capacities = torch.tensor(capacities, dtype=torch.float32).unsqueeze(1)  # (N-2, 1)
    y_targets = torch.tensor(y_targets, dtype=torch.float32)
    return x_pairs, capacities, y_targets

def save_example_to_csv(x_train, y_train, example_idx, filename="ciclo_de_carga.csv"):
    """
    Guarda un ejemplo de x_train con su correspondiente etiqueta de y_train en un archivo CSV.

    Parámetros:
    - x_train: Tensor de entrada con forma (N, 400, 3).
    - y_train: Tensor de etiquetas con forma (N,).
    - example_idx: Índice del ejemplo a guardar.
    - filename: Nombre del archivo CSV de salida (por defecto "example_data_with_label.csv").
    """

    # Verificar que el índice es válido
    if example_idx < 0 or example_idx >= len(x_train):
        raise ValueError(f"Índice fuera de rango: {example_idx}. Debe estar entre 0 y {len(x_train) - 1}.")

    # Aplanar el tensor del ejemplo para convertirlo en un vector 1D de longitud 1200
    example_data = x_train[example_idx].reshape(-1)  # De [400, 3] a [1200]

    # Tomar la etiqueta correspondiente
    example_label = y_train[example_idx]

    # Combinar los datos de entrada con la etiqueta
    data_with_label = np.append(example_data, example_label)  # Unir características y etiqueta

    # Convertir a un DataFrame de pandas
    df = pd.DataFrame(data_with_label.reshape(1, -1))

    # Guardar en un archivo CSV sin encabezados ni índice
    df.to_csv(filename, header=False, index=False)

    print(f"Ejemplo {example_idx} guardado en {filename}")



def save_example_to_csv_narx(x_pair, cap_input, y_target, example_idx, filename="ciclo_de_carga.csv"):
    """
    Guarda un ejemplo de entrada (x_pair + cap_input) y su etiqueta (y_target) en un CSV.

    - x_pair: Tensor con forma (N, 2, 400, 3)
    - cap_input: Tensor con forma (N, 1)
    - y_target: Tensor con forma (N,)
    """

    if example_idx < 0 or example_idx >= len(x_pair):
        raise ValueError(f"Índice fuera de rango: {example_idx}")

    # x_pair: (2, 400, 3) → flatten → 2*400*3 = 2400
    example_data = x_pair[example_idx].reshape(-1).numpy()

    # cap_input: (1,) → float
    example_cap = cap_input[example_idx].numpy()

    # y_target: (1,) → float
    label = y_target[example_idx].numpy()

    # Concatenar todo: [x_pair_flattened, cap_input, label]
    all_data = np.concatenate([example_data, cap_input[example_idx], [label]])

    df = pd.DataFrame(all_data.reshape(1, -1))
    df.to_csv(filename, header=False, index=False)

    print(f"Ejemplo {example_idx} guardado en {filename}")





class ScheduledOptimizer:
    """
    Reference: `jadore801120/attention-is-all-you-need-pytorch \
    <https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py>`_
    """
    def __init__(self, optimizer, d_model: int, warm_up: int) -> None:
        self._optimizer = optimizer
        self.warm_up = warm_up
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> np.array:
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.warm_up, -1.5) * self.n_current_steps
        ])

    def get_lr(self):
        lr = self.init_lr * self._get_lr_scale()
        return lr

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return self._optimizer.state_dict()
