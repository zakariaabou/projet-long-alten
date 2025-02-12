# model_builder.py
import torch
import torch.nn as nn

class NetworkBuilder:
    def __init__(self, input_size, layer_configs, output_size, global_activation="relu"):
        """
        input_size: dimension d'entrée (int)
        layer_configs: liste de dictionnaires, chacun définissant une couche avec les clés :
            - "layer_type" : "Dense" ou "Convolution"
            - "units" : nombre d'unités ou de filtres (int)
            - "kernel_size" : taille du kernel (int) (pour les couches de type Convolution)
            - "activation" : fonction d'activation (str)
        output_size: dimension de sortie (int)
        global_activation: activation par défaut à utiliser (str)
        """
        self.input_size = input_size
        self.layer_configs = layer_configs
        self.output_size = output_size
        self.global_activation = global_activation.lower()

    def build_network(self):
        layers = []
        in_features = self.input_size
        for config in self.layer_configs:
            layer_type = config.get("layer_type", "Dense")
            units = int(config.get("units", 64))
            activation = config.get("activation", self.global_activation)
            if layer_type.lower() == "dense":
                layers.append(nn.Linear(in_features, units))
                layers.append(self._get_activation(activation))
                in_features = units
            elif layer_type.lower() == "convolution":
                # Pour simplifier, ici nous simulons une couche convolutionnelle par une couche linéaire.
                # Dans une implémentation réelle, il faudrait gérer les dimensions 2D, strides, padding, etc.
                layers.append(nn.Linear(in_features, units))
                layers.append(self._get_activation(activation))
                in_features = units
            else:
                raise ValueError("Type de couche non supporté : " + layer_type)
        layers.append(nn.Linear(in_features, self.output_size))
        return nn.Sequential(*layers)

    def _get_activation(self, activation):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.ReLU()  # activation par défaut

def detect_gpu():
    if torch.cuda.is_available():
        nb_gpu = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(nb_gpu)]
        return "GPU(s) détecté(s) : " + ", ".join(gpu_names)
    else:
        return "Aucun GPU détecté, utilisation du CPU."
