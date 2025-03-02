# controller.py
import threading
import torch
from model_builder import NetworkBuilder
from train_manager import Trainer
from data_loader import DataLoader

class GANController:
    def __init__(self, gen_config, disc_config, training_config):
        """
        gen_config: dict contenant la configuration du générateur, par exemple :
            {
                "input_size": 100,
                "layers": [ 
                    {"layer_type": "Dense", "units": 128, "kernel_size": None, "activation": "relu"},
                    {"layer_type": "Dense", "units": 64, "kernel_size": None, "activation": "tanh"}
                ],
                "output_size": 64,
                "global_activation": "relu"
            }
        disc_config: dict similaire pour le discriminateur.
        training_config: dict avec les paramètres d'entraînement, par exemple :
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

        # Extraction des paramètres d'entraînement
        self.gen_train_params = training_config.get("generator", {})
        self.disc_train_params = training_config.get("discriminator", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Initialisation du DataLoader
        self.data_loader = DataLoader(
            data_folder=training_config.get("data_folder", ""),
            batch_size=training_config.get("batch_size", 32),
            image_size=64  # Taille des images (à ajuster selon vos besoins)
        )

        # Initialisation du Trainer avec les deux jeux de paramètres
        self.trainer = Trainer(
            self.generator,
            self.discriminator,
            gen_train_params=self.gen_train_params,
            disc_train_params=self.disc_train_params,
            device=self.device,
            data_loader=self.data_loader
        )

        self.update_learning_rates()
        self.trainer.current_network = training_config.get("initial_network", "générateur")
        self.training_thread_gen = None
        self.training_thread_disc = None

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

    def update_learning_rates(self):
        gen_lr = self.gen_train_params.get("learning_rate", 0.001)
        disc_lr = self.disc_train_params.get("learning_rate", 0.001)
        
        for param_group in self.trainer.gen_optimizer.param_groups:
            param_group['lr'] = gen_lr
        for param_group in self.trainer.disc_optimizer.param_groups:
            param_group['lr'] = disc_lr

    def start_training(self, callback):
        # Vérifier que le dossier de données est valide
        if not self.training_config.get("data_folder"):
            messagebox.showerror("Erreur", "Aucun dossier de données sélectionné.")
            return
        # Récupérer le DataLoader depuis le Trainer
        data_loader = self.data_loader.get_data_loader()
        # Vérifier que le DataLoader a bien chargé des données
        if data_loader is None:
            messagebox.showerror("Erreur", "Impossible de charger les données. Vérifiez le dossier sélectionné.")
            return
        # Récupération des paramètres d'entraînement
        gen_epochs = self.gen_train_params.get("epochs", 10)
        disc_epochs = self.disc_train_params.get("epochs", 10)
        gen_batch_size = self.gen_train_params.get("batch_size", 32)
        disc_batch_size = self.disc_train_params.get("batch_size", 32)
        # Passage des paramètres au trainer
        if self.trainer.current_network == "générateur":
            self.training_thread_gen = threading.Thread(
                target=self.trainer.train_generator,
                args=(
                    gen_epochs,
                    gen_batch_size,
                    callback
                )
            )
            self.training_thread_gen.start()
        else:
            self.training_thread_disc = threading.Thread(
                target=self.trainer.train_discriminator,
                args=(
                    disc_epochs,
                    disc_batch_size,
                    callback
                )
            )
            self.training_thread_disc.start()
        

    def pause_training(self):
        self.trainer.pause()

    def resume_training(self):
        self.trainer.resume()

    def switch_network(self):
        self.trainer.switch()

    def stop_training(self):
        self.trainer.stop()
        if self.training_thread_gen and self.training_thread_gen.is_alive():
            self.training_thread_gen.join()
        if self.training_thread_disc and self.training_thread_disc.is_alive():
            self.training_thread_disc.join()
