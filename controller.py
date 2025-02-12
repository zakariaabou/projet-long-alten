# controller.py
import threading
from model_builder import NetworkBuilder
from train_manager import Trainer
import torch

class GANController:
    def __init__(self, gen_config, disc_config, training_config):
        """
        gen_config: dictionnaire contenant les paramètres du générateur, par exemple :
            {
                "input_size": 100,
                "layers": [ 
                    {"layer_type": "Dense", "units": 128, "kernel_size": None, "activation": "relu"},
                    {"layer_type": "Dense", "units": 64, "kernel_size": None, "activation": "tanh"}
                ],
                "output_size": 64,
                "global_activation": "relu"
            }
        disc_config: dictionnaire similaire pour le discriminateur, par exemple :
            {
                "input_size": 64,
                "layers": [ 
                    {"layer_type": "Dense", "units": 128, "kernel_size": None, "activation": "relu"}
                ],
                "output_size": 1,
                "global_activation": "relu"
            }
        training_config: dictionnaire avec les paramètres d'entraînement, par exemple :
            {
                "learning_rate": 0.001,
                "epochs": 10,
                "batch_size": 32,
                "data_folder": "chemin/vers/donnees",
                "initial_network": "generator"
            }
        """
        self.gen_config = gen_config
        self.disc_config = disc_config
        self.training_config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Construction des modèles
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        # Création du Trainer
        self.trainer = Trainer(self.generator, self.discriminator, device=self.device)
        self.update_learning_rates(training_config.get("learning_rate", 0.001))
        self.trainer.current_network = training_config.get("initial_network", "generator")
        self.training_thread = None

    def build_generator(self):
        input_size = self.gen_config.get("input_size", 100)
        layers = self.gen_config.get("layers", [])
        output_size = self.gen_config.get("output_size", 64)
        global_activation = self.gen_config.get("global_activation", "relu")
        builder = NetworkBuilder(input_size, layers, output_size, global_activation)
        return builder.build_network()

    def build_discriminator(self):
        input_size = self.disc_config.get("input_size", 64)
        layers = self.disc_config.get("layers", [])
        output_size = self.disc_config.get("output_size", 1)
        global_activation = self.disc_config.get("global_activation", "relu")
        builder = NetworkBuilder(input_size, layers, output_size, global_activation)
        return builder.build_network()

    def update_learning_rates(self, lr):
        for param_group in self.trainer.gen_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.trainer.disc_optimizer.param_groups:
            param_group['lr'] = lr

    def start_training(self, callback):
        epochs = self.training_config.get("epochs", 10)
        batch_size = self.training_config.get("batch_size", 32)
        self.training_thread = threading.Thread(target=self.trainer.train, args=(epochs, batch_size, callback))
        self.training_thread.start()

    def pause_training(self):
        self.trainer.pause()

    def resume_training(self):
        self.trainer.resume()

    def switch_network(self):
        self.trainer.switch()

    def stop_training(self):
        self.trainer.stop()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join()
