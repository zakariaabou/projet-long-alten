import tkinter as tk
from tkinter import ttk, messagebox

class GanConfigurator:
    def __init__(self, root):
        self.root = root
        self.root.title("Configuration du Réseau GAN")
        self.root.geometry("800x600")
        
        # Création d'un Notebook pour organiser les onglets
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Création des différents onglets
        self.gen_frame = ttk.Frame(self.notebook)
        self.disc_frame = ttk.Frame(self.notebook)
        self.train_frame = ttk.Frame(self.notebook)
        self.summary_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.gen_frame, text="Générateur")
        self.notebook.add(self.disc_frame, text="Discriminateur")
        self.notebook.add(self.train_frame, text="Paramètres d'Entraînement")
        self.notebook.add(self.summary_frame, text="Résumé de la Configuration")

        # Initialisation des interfaces de chaque onglet
        self.build_generator_tab()
        self.build_discriminator_tab()
        self.build_training_tab()
        self.build_summary_tab()

    # ---------------------- Onglet Générateur ----------------------
    def build_generator_tab(self):
        frame = self.gen_frame
        ttk.Label(frame, text="Configuration du Générateur", font=("Arial", 14)).pack(pady=10)

        # Conteneur pour la configuration des couches
        self.gen_layers_container = ttk.Frame(frame)
        self.gen_layers_container.pack(fill="both", expand=True, padx=10, pady=10)

        # En-tête des colonnes
        header_frame = ttk.Frame(self.gen_layers_container)
        header_frame.pack(fill="x")
        ttk.Label(header_frame, text="Type de couche", width=20).grid(row=0, column=0)
        ttk.Label(header_frame, text="Units/Filters", width=20).grid(row=0, column=1)
        ttk.Label(header_frame, text="Kernel Size", width=20).grid(row=0, column=2)

        # Liste pour stocker les lignes de configuration
        self.gen_layer_rows = []

        # Bouton pour ajouter une nouvelle couche
        add_layer_btn = ttk.Button(frame, text="Ajouter une couche", command=self.add_gen_layer)
        add_layer_btn.pack(pady=5)

        # Choix de la fonction d'activation
        activation_frame = ttk.Frame(frame)
        activation_frame.pack(pady=5)
        ttk.Label(activation_frame, text="Fonction d'activation :").pack(side="left")
        self.gen_activation = ttk.Combobox(activation_frame, values=["relu", "tanh", "sigmoid"], width=10)
        self.gen_activation.set("relu")
        self.gen_activation.pack(side="left", padx=5)

        # Bouton de prévisualisation et zone de résumé
        preview_btn = ttk.Button(frame, text="Prévisualiser Générateur", command=self.preview_generator)
        preview_btn.pack(pady=5)
        self.gen_summary_text = tk.Text(frame, height=6, width=80)
        self.gen_summary_text.pack(pady=5)

    def add_gen_layer(self):
        # Création d'une ligne pour une couche du générateur
        row_frame = ttk.Frame(self.gen_layers_container)
        row_frame.pack(fill="x", pady=2)
        # Dropdown pour le type de couche
        layer_type_cb = ttk.Combobox(row_frame, values=["Dense", "Convolution"], width=18)
        layer_type_cb.set("Dense")
        layer_type_cb.grid(row=0, column=0, padx=5)
        # Champ pour le nombre d'unités ou filtres
        units_entry = ttk.Entry(row_frame, width=20)
        units_entry.grid(row=0, column=1, padx=5)
        # Champ pour la taille du kernel (utile pour Convolution)
        kernel_entry = ttk.Entry(row_frame, width=20)
        kernel_entry.grid(row=0, column=2, padx=5)
        # Bouton pour supprimer cette ligne
        remove_btn = ttk.Button(row_frame, text="Supprimer", command=lambda: self.remove_gen_layer(row_frame))
        remove_btn.grid(row=0, column=3, padx=5)
        self.gen_layer_rows.append((row_frame, layer_type_cb, units_entry, kernel_entry))

    def remove_gen_layer(self, row_frame):
        for i, (frame, _, _, _) in enumerate(self.gen_layer_rows):
            if frame == row_frame:
                frame.destroy()
                del self.gen_layer_rows[i]
                break

    def preview_generator(self):
        summary = "Générateur:\n"
        for (frame, layer_type_cb, units_entry, kernel_entry) in self.gen_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            summary += f"- {layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
        activation = self.gen_activation.get()
        summary += f"Activation : {activation}\n"
        self.gen_summary_text.delete("1.0", tk.END)
        self.gen_summary_text.insert(tk.END, summary)

    # ---------------------- Onglet Discriminateur ----------------------
    def build_discriminator_tab(self):
        frame = self.disc_frame
        ttk.Label(frame, text="Configuration du Discriminateur", font=("Arial", 14)).pack(pady=10)

        self.disc_layers_container = ttk.Frame(frame)
        self.disc_layers_container.pack(fill="both", expand=True, padx=10, pady=10)

        header_frame = ttk.Frame(self.disc_layers_container)
        header_frame.pack(fill="x")
        ttk.Label(header_frame, text="Type de couche", width=20).grid(row=0, column=0)
        ttk.Label(header_frame, text="Units/Filters", width=20).grid(row=0, column=1)
        ttk.Label(header_frame, text="Kernel Size", width=20).grid(row=0, column=2)

        self.disc_layer_rows = []
        add_layer_btn = ttk.Button(frame, text="Ajouter une couche", command=self.add_disc_layer)
        add_layer_btn.pack(pady=5)

        activation_frame = ttk.Frame(frame)
        activation_frame.pack(pady=5)
        ttk.Label(activation_frame, text="Fonction d'activation :").pack(side="left")
        self.disc_activation = ttk.Combobox(activation_frame, values=["relu", "tanh", "sigmoid"], width=10)
        self.disc_activation.set("relu")
        self.disc_activation.pack(side="left", padx=5)

        preview_btn = ttk.Button(frame, text="Prévisualiser Discriminateur", command=self.preview_discriminator)
        preview_btn.pack(pady=5)
        self.disc_summary_text = tk.Text(frame, height=6, width=80)
        self.disc_summary_text.pack(pady=5)

    def add_disc_layer(self):
        row_frame = ttk.Frame(self.disc_layers_container)
        row_frame.pack(fill="x", pady=2)
        layer_type_cb = ttk.Combobox(row_frame, values=["Dense", "Convolution"], width=18)
        layer_type_cb.set("Dense")
        layer_type_cb.grid(row=0, column=0, padx=5)
        units_entry = ttk.Entry(row_frame, width=20)
        units_entry.grid(row=0, column=1, padx=5)
        kernel_entry = ttk.Entry(row_frame, width=20)
        kernel_entry.grid(row=0, column=2, padx=5)
        remove_btn = ttk.Button(row_frame, text="Supprimer", command=lambda: self.remove_disc_layer(row_frame))
        remove_btn.grid(row=0, column=3, padx=5)
        self.disc_layer_rows.append((row_frame, layer_type_cb, units_entry, kernel_entry))

    def remove_disc_layer(self, row_frame):
        for i, (frame, _, _, _) in enumerate(self.disc_layer_rows):
            if frame == row_frame:
                frame.destroy()
                del self.disc_layer_rows[i]
                break

    def preview_discriminator(self):
        summary = "Discriminateur:\n"
        for (frame, layer_type_cb, units_entry, kernel_entry) in self.disc_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            summary += f"- {layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
        activation = self.disc_activation.get()
        summary += f"Activation : {activation}\n"
        self.disc_summary_text.delete("1.0", tk.END)
        self.disc_summary_text.insert(tk.END, summary)

    # ---------------------- Onglet Paramètres d'Entraînement ----------------------
    def build_training_tab(self):
        frame = self.train_frame
        ttk.Label(frame, text="Paramètres d'Entraînement", font=("Arial", 14)).pack(pady=10)

        # Choix du réseau à entraîner
        ttk.Label(frame, text="Réseau à entraîner :").pack(pady=5)
        self.train_choice = tk.StringVar(value="Générateur")
        choice_frame = ttk.Frame(frame)
        choice_frame.pack()
        ttk.Radiobutton(choice_frame, text="Générateur", variable=self.train_choice, value="Générateur").pack(side="left", padx=10)
        ttk.Radiobutton(choice_frame, text="Discriminateur", variable=self.train_choice, value="Discriminateur").pack(side="left", padx=10)

        # Sélection de la fonction de perte
        ttk.Label(frame, text="Fonction de perte :").pack(pady=5)
        self.loss_function = ttk.Combobox(frame, values=["MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss"], width=15)
        self.loss_function.set("MSELoss")
        self.loss_function.pack(pady=5)

        # Autres paramètres d'entraînement
        ttk.Label(frame, text="Learning Rate :").pack(pady=5)
        self.learning_rate_entry = ttk.Entry(frame, width=10)
        self.learning_rate_entry.insert(0, "0.001")
        self.learning_rate_entry.pack(pady=5)

        ttk.Label(frame, text="Nombre d'epochs :").pack(pady=5)
        self.epochs_entry = ttk.Entry(frame, width=10)
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.pack(pady=5)

        ttk.Label(frame, text="Taille du batch :").pack(pady=5)
        self.batch_size_entry = ttk.Entry(frame, width=10)
        self.batch_size_entry.insert(0, "32")
        self.batch_size_entry.pack(pady=5)

        # Boutons de contrôle d'entraînement
        control_frame = ttk.Frame(frame)
        control_frame.pack(pady=10)
        pause_btn = ttk.Button(control_frame, text="Pause Entraînement", command=self.pause_training)
        pause_btn.pack(side="left", padx=5)
        resume_btn = ttk.Button(control_frame, text="Reprendre Entraînement", command=self.resume_training)
        resume_btn.pack(side="left", padx=5)
        switch_btn = ttk.Button(control_frame, text="Switcher Réseau", command=self.switch_training)
        switch_btn.pack(side="left", padx=5)

        # Zone d'affichage des statistiques d'entraînement
        self.training_stats_text = tk.Text(frame, height=6, width=80)
        self.training_stats_text.pack(pady=5)

    def pause_training(self):
        self.training_stats_text.delete("1.0", tk.END)
        self.training_stats_text.insert(tk.END, "Entraînement mis en pause.\n")

    def resume_training(self):
        self.training_stats_text.delete("1.0", tk.END)
        self.training_stats_text.insert(tk.END, "Entraînement repris.\n")

    def switch_training(self):
        self.training_stats_text.delete("1.0", tk.END)
        self.training_stats_text.insert(tk.END, "Switch effectué : changement du réseau à entraîner.\n")

    # ---------------------- Onglet Résumé de la Configuration ----------------------
    def build_summary_tab(self):
        frame = self.summary_frame
        ttk.Label(frame, text="Résumé de la Configuration", font=("Arial", 14)).pack(pady=10)
        self.summary_text = tk.Text(frame, height=15, width=80)
        self.summary_text.pack(pady=10)
        summary_btn = ttk.Button(frame, text="Générer le Résumé", command=self.generate_summary)
        summary_btn.pack(pady=5)

    def generate_summary(self):
        summary = "=== Générateur ===\n"
        for (frame, layer_type_cb, units_entry, kernel_entry) in self.gen_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            summary += f"{layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
        summary += f"Activation Générateur: {self.gen_activation.get()}\n\n"
        summary += "=== Discriminateur ===\n"
        for (frame, layer_type_cb, units_entry, kernel_entry) in self.disc_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            summary += f"{layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
        summary += f"Activation Discriminateur: {self.disc_activation.get()}\n\n"
        summary += "=== Paramètres d'Entraînement ===\n"
        summary += f"Réseau à entraîner: {self.train_choice.get()}\n"
        summary += f"Fonction de perte: {self.loss_function.get()}\n"
        summary += f"Learning Rate: {self.learning_rate_entry.get()}\n"
        summary += f"Nombre d'epochs: {self.epochs_entry.get()}\n"
        summary += f"Taille du batch: {self.batch_size_entry.get()}\n"
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)

if __name__ == "__main__":
    root = tk.Tk()
    app = GanConfigurator(root)
    root.mainloop()
