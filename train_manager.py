# train_manager.py
import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, generator, discriminator, gen_train_params, disc_train_params, device=None, data_loader=None):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configuration d'entraînement séparée
        self.gen_train_params = gen_train_params
        self.disc_train_params = disc_train_params
        
        # Initialisation des réseaux
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Optimizers avec learning rates séparés
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=gen_train_params.get('learning_rate', 0.001)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=disc_train_params.get('learning_rate', 0.001)
        )
        
        # Fonctions de perte séparées
        self.gen_loss_fn = self._get_loss_function(gen_train_params.get('loss_function', 'BCELoss'))
        self.disc_loss_fn = self._get_loss_function(disc_train_params.get('loss_function', 'BCELoss'))
        
        # État d'entraînement
        self.running_gen = False
        self.running_disc = False
        self.pause_event_gen = threading.Event()
        self.pause_event_gen.set()
        self.pause_event_disc = threading.Event()
        self.pause_event_disc.set()
        self.current_network = "générateur"
        self.epoch = 0
        self.data_loader = data_loader

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def _get_loss_function(self, loss_name):
        loss_dict = {
            "MSELoss": nn.MSELoss(),
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
            "CrossEntropyLoss": nn.CrossEntropyLoss(),
            "BCELoss": nn.BCELoss()
        }
        return loss_dict.get(loss_name, nn.BCELoss())
    
    def train_generator(self, epochs, batch_size, callback=None):
        self.running_gen = True
        self.current_network = "générateur"
        for epoch in range(1, epochs + 1):
            if not self.running_gen:
                break
            self.pause_event_gen.wait()
            for i in range(3):
                noise = torch.randn(batch_size, 98).to(self.device) #self.generator[-1].out_features
                
                self._freeze(self.discriminator, True)
                self._freeze(self.generator, False)
                loss = self._train_generator(noise)
            msg = f"[Générateur] Epoch {epoch}/{epochs} - Loss: {loss:.4f}"
            time.sleep(0.5)
            if callback:
                callback(msg)
        self.running_gen = False
    
    def train_discriminator(self, epochs, batch_size, callback=None):
        self.running_disc = True
        self.current_network = "discriminateur"
        data_loader = self.data_loader.get_data_loader()
        for epoch in range(1, epochs + 1):
            if not self.running_disc:
                break
            self.pause_event_disc.wait()
            for i, (real_data, _) in enumerate(data_loader):
                real_data = real_data.to(self.device)
                noise = torch.randn(batch_size, 3136).to(self.device)
                fake_data = self.generator(noise).detach()
                #self.show_generated_images(fake_data)
                
                self._freeze(self.generator, True)
                self._freeze(self.discriminator, False)
                loss = self._train_discriminator(real_data, fake_data)
            msg = f"[Discriminateur] Epoch {epoch}/{epochs} - Loss: {loss:.4f}"
            time.sleep(0.5)
            if callback:
                callback(msg)
        self.running_disc = False

    def _train_discriminator(self, real_data, fake_data):
        self.disc_optimizer.zero_grad()
        
        real_labels = torch.ones(real_data.size(0), 1).to(self.device)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(self.device)
        
        # Calcul des pertes
        output_real = self.discriminator(real_data)
        loss_real = self.disc_loss_fn(output_real, real_labels)
        output_fake = self.discriminator(fake_data)
        loss_fake = self.disc_loss_fn(output_fake, fake_labels)
        
        loss = loss_real + loss_fake
        loss.backward()
        self.disc_optimizer.step()
        return loss.item()

    def _train_generator(self, noise):
        self.gen_optimizer.zero_grad()
        
        fake_data = self.generator(noise)
        #.show_generated_images(fake_data)
        fake_labels = torch.ones(fake_data.size(0), 1).to(self.device)
        
        # Calcul de la perte
        output = self.discriminator(fake_data)
        loss = self.gen_loss_fn(output, fake_labels)
        
        loss.backward()
        self.gen_optimizer.step()
        return loss.item()

    def _freeze(self, model, freeze=True):
        for param in model.parameters():
            param.requires_grad = not freeze

    def pause(self):
        self.pause_event_gen.clear()
        self.pause_event_disc.clear()


    def resume(self):
        self.pause_event_gen.set() if self.current_network == "générateur" else self.pause_event_disc.set()

    def stop(self):
        self.running_disc = False
        self.running_gen = False

    def switch(self):
        if self.current_network == "générateur":
            self.current_network = "discriminator"
            if self.pause_event_gen.is_set():
                self.pause_event_gen.clear()
                self.pause_event_disc.set()
        else:
            self.current_network = "générateur"
            if self.pause_event_disc.is_set():
                self.pause_event_disc.clear()
                self.pause_event_gen.set()
    
    def plot_losses(self, gen_losses, disc_losses):
        plt.figure(figsize=(12, 6))
        plt.plot(gen_losses, label="Générateur")
        plt.plot(disc_losses, label="Discriminateur")
        plt.legend()
        plt.title("Évolution des pertes")
        plt.show()
    
    def save_model(self, generator_path, discriminator_path):
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def load_model(self, generator_path, discriminator_path):
        self.generator.load_state_dict(torch.load(generator_path))
        self.discriminator.load_state_dict(torch.load(discriminator_path))

    def show_generated_images(self, images, num_images=5):
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for i in range(num_images):
            axes[i].imshow(images[i].permute(1, 2, 0).cpu().detach().numpy() * 0.5 + 0.5)
            axes[i].axis('off')
        plt.show()
    
    def save_images(self, path, num_images=5):
        noise = torch.randn(num_images, self.generator[-1].out_features).to(self.device)
        fake_images = self.generator(noise)
        self.show_generated_images(fake_images, num_images)
        plt.savefig(path)
