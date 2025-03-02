# model_builder.py
import torch
import torch.nn as nn

class NetworkBuilder:
    def __init__(self, input_size, layer_configs, output_size, global_activation="relu", input_channels=3):
        """
        input_size: dimension d'entrée (int pour couches denses) ou tuple (hauteur, largeur) pour convolutions
        layer_configs: liste de configurations de couches
        output_size: dimension de sortie
        global_activation: activation par défaut
        input_channels: nombre de canaux d'entrée (défaut: 3)
        """
        self.input_size = input_size
        self.layer_configs = layer_configs
        self.output_size = output_size
        self.global_activation = global_activation.lower()
        self.input_channels = input_channels

    def build_network(self):
        layers = []
        current_shape = self._get_initial_shape()

        for config in self.layer_configs:
            config = {k.lower(): v for k, v in config.items()}
            layer_type = config.get("layer_type", "dense").lower()
            activation = config.get("activation", self.global_activation).lower()

            if layer_type == "dense":
                current_shape = self._add_dense_layer(layers, config, current_shape)
            elif layer_type == "convolution":
                current_shape = self._add_conv_layer(layers, config, current_shape)
            elif layer_type == "transposed_conv":
                current_shape = self._add_transposed_conv_layer(layers, config, current_shape)
            elif layer_type in ["maxpool", "avgpool"]:
                current_shape = self._add_pool_layer(layers, config, current_shape)
            elif layer_type == "flatten":
                current_shape = self._add_flatten(layers, current_shape)
            elif layer_type == "batchnorm":
                layers.append(nn.BatchNorm1d(current_shape[0]))
            elif layer_type == "dropout":
                layers.append(nn.Dropout(config.get("probability", 0.5)))
            elif layer_type == "upsample":
                current_shape = self._add_upsample(layers, config, current_shape)
            elif layer_type == "unflatten":
                current_shape = self._add_unflatten(layers, config, current_shape)
            else:
                raise ValueError(f"Type de couche non supporté : {layer_type}")

            if "activation" in config:
                layers.append(self._get_activation(activation))

        # Ajout de la couche de sortie
        layers.append(self._add_output_layer(current_shape))
        return nn.Sequential(*layers)

    def _get_initial_shape(self):
        if isinstance(self.input_size, tuple):
            return (self.input_channels, self.input_size[0], self.input_size[1])
        return (self.input_size,)

    def _add_dense_layer(self, layers, config, current_shape):
        if len(current_shape) > 1:
            raise ValueError("Les couches Dense nécessitent une entrée plate. Ajoutez une couche Flatten d'abord.")
        
        in_features = current_shape[0]
        out_features = config.get("units", 64)
        layers.append(nn.Linear(in_features, out_features))
        return (out_features,)

    def _add_conv_layer(self, layers, config, current_shape):
        if len(current_shape) == 1:
            raise ValueError("Les couches de convolution nécessitent une entrée 2D. Ajoutez une couche Unflatten d'abord.")
        in_channels = current_shape[0]
        out_channels = config.get("units", 64)
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 1)
        padding = config.get("padding", 0)

        layers.append(nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))

        # Calcul nouvelle forme
        h = (current_shape[1] - kernel_size + 2 * padding) // stride + 1
        w = (current_shape[2] - kernel_size + 2 * padding) // stride + 1
        return (out_channels, h, w)
    
    def _add_transposed_conv_layer(self, layers, config, current_shape):
        if len(current_shape) == 1:
            raise ValueError("Les couches de convolution nécessitent une entrée 2D. Ajoutez une couche Unflatten d'abord.")
        in_channels = current_shape[0]
        out_channels = config.get("units", 64)
        kernel_size = config.get("kernel_size", 4)
        stride = config.get("stride", 2)
        padding = config.get("padding", 1)
        
        layers.append(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        ))
        layers.append(nn.BatchNorm2d(out_channels))
        
        h = (current_shape[1] - 1) * stride - 2 * padding + kernel_size
        w = (current_shape[2] - 1) * stride - 2 * padding + kernel_size
        return (out_channels, h, w)

    def _add_pool_layer(self, layers, config, current_shape):
        kernel_size = config.get("kernel_size", 2)
        stride = config.get("stride", 2)
        
        if config["layer_type"] == "maxpool":
            layers.append(nn.MaxPool2d(kernel_size, stride))
        else:
            layers.append(nn.AvgPool2d(kernel_size, stride))

        h = (current_shape[1] - kernel_size) // stride + 1
        w = (current_shape[2] - kernel_size) // stride + 1
        return (current_shape[0], h, w)

    def _add_flatten(self, layers, current_shape):
        layers.append(nn.Flatten())
        return (current_shape[0] * current_shape[1] * current_shape[2],)

    def _add_upsample(self, layers, config, current_shape):
        scale_factor = config.get("scale_factor", 2)
        mode = config.get("mode", "nearest")
        layers.append(nn.Upsample(scale_factor=scale_factor, mode=mode))
        return (
            current_shape[0],
            current_shape[1] * scale_factor,
            current_shape[2] * scale_factor
        )

    def _add_output_layer(self, current_shape):
        if len(current_shape) == 1:  # Dense
            return nn.Linear(current_shape[0], self.output_size)
        else:  # Conv
            return nn.Conv2d(current_shape[0], self.output_size, kernel_size=1)
        
    def _add_unflatten(self, layers, config, current_shape):
        h, w = config.get("height", 7), config.get("width", 7)
        c = current_shape[0] // (h * w)
        layers.append(nn.Unflatten(1, (c, h, w)))
        return (c, h, w)

    def _get_activation(self, activation):
        activations = {
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.2),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1)
        }
        return activations.get(activation, nn.ReLU())

def detect_gpu():
    if torch.cuda.is_available():
        return f"{torch.cuda.device_count()} GPU(s) - {torch.cuda.get_device_name(0)}"
    return "Aucun GPU détecté"
