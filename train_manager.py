# train_manager.py
import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time

class Trainer:
    def __init__(self, generator, discriminator, device=None):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.running = False
        self.pause_event = threading.Event()
        self.pause_event.set()  # pas de pause au début
        self.current_network = "generator"  # peut être "generator" ou "discriminator"
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.loss_fn = nn.BCELoss()  # Exemple avec une loss binaire
        self.epoch = 0

    def train(self, epochs, batch_size, callback=None):
        self.running = True
        for epoch in range(1, epochs + 1):
            if not self.running:
                break
            self.epoch = epoch
            # Attente en cas de pause
            self.pause_event.wait()
            
            # Simulation d'une itération d'entraînement
            noise = torch.randn(batch_size, 100).to(self.device)
            # Simulation de données réelles fictives
            real_data = torch.randn(batch_size, self.generator[-1].out_features).to(self.device)
            
            if self.current_network == "discriminator":
                self._freeze(self.generator, True)
                self._freeze(self.discriminator, False)
                fake_data = self.generator(noise).detach()
                loss_disc = self._train_discriminator(real_data, fake_data)
                msg = f"Epoch {epoch}: Discriminateur Loss = {loss_disc:.4f}"
            else:
                self._freeze(self.discriminator, True)
                self._freeze(self.generator, False)
                loss_gen = self._train_generator(noise)
                msg = f"Epoch {epoch}: Générateur Loss = {loss_gen:.4f}"
            
            time.sleep(0.5)  # pause pour simuler le temps d'entraînement
            if callback:
                callback(msg)
        self.running = False

    def _train_discriminator(self, real_data, fake_data):
        self.disc_optimizer.zero_grad()
        real_labels = torch.ones(real_data.size(0), 1).to(self.device)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(self.device)
        output_real = self.discriminator(real_data)
        loss_real = self.loss_fn(output_real, real_labels)
        output_fake = self.discriminator(fake_data)
        loss_fake = self.loss_fn(output_fake, fake_labels)
        loss = loss_real + loss_fake
        loss.backward()
        self.disc_optimizer.step()
        return loss.item()

    def _train_generator(self, noise):
        self.gen_optimizer.zero_grad()
        fake_data = self.generator(noise)
        fake_labels = torch.ones(fake_data.size(0), 1).to(self.device)
        output = self.discriminator(fake_data)
        loss = self.loss_fn(output, fake_labels)
        loss.backward()
        self.gen_optimizer.step()
        return loss.item()

    def _freeze(self, model, freeze=True):
        for param in model.parameters():
            param.requires_grad = not freeze

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def stop(self):
        self.running = False

    def switch(self):
        if self.current_network == "generator":
            self.current_network = "discriminator"
        else:
            self.current_network = "generator"
