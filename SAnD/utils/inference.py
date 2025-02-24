import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch

from SAnD.core.model import SAnD, SAnD_Embedding, SiameseSAnD

class Inference_SoH:
    def __init__(self, model_path, input_features, seq_len, n_heads, factor, n_class, n_layers, device="cuda"):
        self.device = device
        self.siamese_model = SiameseSAnD(SAnD_Embedding(input_features, seq_len, n_heads, factor, n_class, n_layers))
        checkpoint = torch.load("save_params/trained_model_siames.pth", map_location=device)
        self.siamese_model.load_state_dict(checkpoint["model_state_dict"])
        #self.siamese_model.load_state_dict(torch.load(model_path, map_location=device))
        self.siamese_model.to(device)
        self.siamese_model.eval()

        self.sand_model = SAnD(input_features, seq_len, n_heads, factor, n_class, n_layers)
        self.sand_model.encoder.load_state_dict(self.siamese_model.sand.encoder.state_dict())
        self.sand_model.to(device)
        self.sand_model.eval()

    def predict(self, test_loader):
        predictions = []
        soh_real = []
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.clone().detach().to(self.device)
                x_test = x_test.to(self.device)
                soh_raw = self.sand_model(x_test)  # Obtener SoH
                soh_pred = torch.sigmoid(soh_raw).cpu().numpy()
                predictions.append(soh_pred)
                soh_real.append(y_test.cpu().numpy())
            # ##########################################################################
        # # PLoteo de la inferencia
        # Convertir listas a numpy arrays
        soh_pred = np.concatenate(predictions).flatten()
        soh_real = np.concatenate(soh_real).flatten()

        # Llamar a la función de visualización
        self.plot_soh(soh_real, soh_pred)

        return soh_pred

    def plot_soh(self, soh_real, soh_pred):
        """Genera un gráfico comparando SoH real vs. SoH predicho."""
        ciclos = np.arange(len(soh_real))

        plt.figure(figsize=(10, 5))
        plt.plot(ciclos, soh_real, marker='o', linestyle='-', color='blue', label='SoH Real')
        plt.plot(ciclos, soh_pred, marker='s', linestyle='--', color='red', label='SoH Predicho')

        plt.xlabel("Ciclo de carga")
        plt.ylabel("State of Health (SoH)")
        plt.title("Comparación de SoH Real vs. SoH Predicho")
        plt.legend()
        plt.grid(True)
        plt.show()
        # ##########################################################################
