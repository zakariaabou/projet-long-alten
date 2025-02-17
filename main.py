# main.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from controller import GANController
from model_builder import detect_gpu

class GanConfigurator:
    def __init__(self, root):
        self.root = root
        self.root.title("Application de Formation de GAN")
        self.root.geometry("1000x700")
        
        # Variable pour stocker le dossier de données
        self.data_folder = tk.StringVar(value="Aucun dossier sélectionné")
        
        # Instance du contrôleur GAN (sera créée lors du démarrage)
        self.gan_controller = None
        
        # Création du Notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        
        # Onglets
        self.gen_frame = ttk.Frame(self.notebook)
        self.disc_frame = ttk.Frame(self.notebook)
        self.train_frame = ttk.Frame(self.notebook)
        self.summary_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.gen_frame, text="Générateur")
        self.notebook.add(self.disc_frame, text="Discriminateur")
        self.notebook.add(self.train_frame, text="Paramètres d'Entraînement")
        self.notebook.add(self.summary_frame, text="Résumé de la Configuration")
        
        self.build_generator_tab()
        self.build_discriminator_tab()
        self.build_training_tab()
        self.build_summary_tab()
        
    # ---------------------- Onglet Générateur ----------------------
    def build_generator_tab(self):
        frame = self.gen_frame
        ttk.Label(frame, text="Configuration du Générateur", font=("Arial", 14)).pack(pady=10)
        
        self.gen_layers_container = ttk.Frame(frame)
        self.gen_layers_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        header_frame = ttk.Frame(self.gen_layers_container)
        header_frame.pack(fill="x")
        ttk.Label(header_frame, text="Type de couche", width=15).grid(row=0, column=0)
        ttk.Label(header_frame, text="Units/Filters", width=15).grid(row=0, column=1)
        ttk.Label(header_frame, text="Kernel Size", width=15).grid(row=0, column=2)
        ttk.Label(header_frame, text="Activation", width=15).grid(row=0, column=3)
        
        self.gen_layer_rows = []
        add_layer_btn = ttk.Button(frame, text="Ajouter une couche", command=self.add_gen_layer)
        add_layer_btn.pack(pady=5)
        
        activation_frame = ttk.Frame(frame)
        activation_frame.pack(pady=5)
        ttk.Label(activation_frame, text="Activation globale (optionnel) :").pack(side="left")
        self.gen_global_activation = ttk.Combobox(activation_frame, values=["relu", "tanh", "sigmoid"], width=10)
        self.gen_global_activation.set("relu")
        self.gen_global_activation.pack(side="left", padx=5)
        
        preview_btn = ttk.Button(frame, text="Prévisualiser Générateur", command=self.preview_generator)
        preview_btn.pack(pady=5)
        self.gen_summary_text = tk.Text(frame, height=8, width=80)
        self.gen_summary_text.pack(pady=5)
        
    def add_gen_layer(self):
        row_frame = ttk.Frame(self.gen_layers_container)
        row_frame.pack(fill="x", pady=2)
        layer_type_cb = ttk.Combobox(row_frame, values=["Dense", "Convolution"], width=13)
        layer_type_cb.set("Dense")
        layer_type_cb.grid(row=0, column=0, padx=5)
        units_entry = ttk.Entry(row_frame, width=15)
        units_entry.grid(row=0, column=1, padx=5)
        kernel_entry = ttk.Entry(row_frame, width=15)
        kernel_entry.grid(row=0, column=2, padx=5)
        activation_cb = ttk.Combobox(row_frame, values=["relu", "tanh", "sigmoid"], width=13)
        activation_cb.set("relu")
        activation_cb.grid(row=0, column=3, padx=5)
        remove_btn = ttk.Button(row_frame, text="Supprimer", command=lambda: self.remove_gen_layer(row_frame))
        remove_btn.grid(row=0, column=4, padx=5)
        self.gen_layer_rows.append((row_frame, layer_type_cb, units_entry, kernel_entry, activation_cb))
    
    def remove_gen_layer(self, row_frame):
        for i, (frame, _, _, _, _) in enumerate(self.gen_layer_rows):
            if frame == row_frame:
                frame.destroy()
                del self.gen_layer_rows[i]
                break
                
    def preview_generator(self):
        summary = "Générateur:\n"
        for (_, layer_type_cb, units_entry, kernel_entry, activation_cb) in self.gen_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            layer_activation = activation_cb.get()
            summary += f"- {layer_type}: Units/Filters = {units}, Kernel Size = {kernel}, Activation = {layer_activation}\n"
        global_act = self.gen_global_activation.get()
        summary += f"Activation globale (optionnel) : {global_act}\n"
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
        ttk.Label(header_frame, text="Type de couche", width=15).grid(row=0, column=0)
        ttk.Label(header_frame, text="Units/Filters", width=15).grid(row=0, column=1)
        ttk.Label(header_frame, text="Kernel Size", width=15).grid(row=0, column=2)
        ttk.Label(header_frame, text="Activation", width=15).grid(row=0, column=3)
        
        self.disc_layer_rows = []
        add_layer_btn = ttk.Button(frame, text="Ajouter une couche", command=self.add_disc_layer)
        add_layer_btn.pack(pady=5)
        
        activation_frame = ttk.Frame(frame)
        activation_frame.pack(pady=5)
        ttk.Label(activation_frame, text="Activation globale (optionnel) :").pack(side="left")
        self.disc_global_activation = ttk.Combobox(activation_frame, values=["relu", "tanh", "sigmoid"], width=10)
        self.disc_global_activation.set("relu")
        self.disc_global_activation.pack(side="left", padx=5)
        
        preview_btn = ttk.Button(frame, text="Prévisualiser Discriminateur", command=self.preview_discriminator)
        preview_btn.pack(pady=5)
        self.disc_summary_text = tk.Text(frame, height=8, width=80)
        self.disc_summary_text.pack(pady=5)
        
    def add_disc_layer(self):
        row_frame = ttk.Frame(self.disc_layers_container)
        row_frame.pack(fill="x", pady=2)
        layer_type_cb = ttk.Combobox(row_frame, values=["Dense", "Convolution"], width=13)
        layer_type_cb.set("Dense")
        layer_type_cb.grid(row=0, column=0, padx=5)
        units_entry = ttk.Entry(row_frame, width=15)
        units_entry.grid(row=0, column=1, padx=5)
        kernel_entry = ttk.Entry(row_frame, width=15)
        kernel_entry.grid(row=0, column=2, padx=5)
        activation_cb = ttk.Combobox(row_frame, values=["relu", "tanh", "sigmoid"], width=13)
        activation_cb.set("relu")
        activation_cb.grid(row=0, column=3, padx=5)
        remove_btn = ttk.Button(row_frame, text="Supprimer", command=lambda: self.remove_disc_layer(row_frame))
        remove_btn.grid(row=0, column=4, padx=5)
        self.disc_layer_rows.append((row_frame, layer_type_cb, units_entry, kernel_entry, activation_cb))
    
    def remove_disc_layer(self, row_frame):
        for i, (frame, _, _, _, _) in enumerate(self.disc_layer_rows):
            if frame == row_frame:
                frame.destroy()
                del self.disc_layer_rows[i]
                break
    
    def preview_discriminator(self):
        summary = "Discriminateur:\n"
        for (_, layer_type_cb, units_entry, kernel_entry, activation_cb) in self.disc_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            layer_activation = activation_cb.get()
            summary += f"- {layer_type}: Units/Filters = {units}, Kernel Size = {kernel}, Activation = {layer_activation}\n"
        global_act = self.disc_global_activation.get()
        summary += f"Activation globale (optionnel) : {global_act}\n"
        self.disc_summary_text.delete("1.0", tk.END)
        self.disc_summary_text.insert(tk.END, summary)
    
    # ---------------------- Onglet Paramètres d'Entraînement ----------------------
    def build_training_tab(self):
        frame = self.train_frame
        ttk.Label(frame, text="Paramètres d'Entraînement", font=("Arial", 14)).pack(pady=10)
        
        ttk.Label(frame, text="Réseau à entraîner :").pack(pady=5)
        self.train_choice = tk.StringVar(value="Générateur")
        choice_frame = ttk.Frame(frame)
        choice_frame.pack()
        ttk.Radiobutton(choice_frame, text="Générateur", variable=self.train_choice, value="Générateur").pack(side="left", padx=10)
        ttk.Radiobutton(choice_frame, text="Discriminateur", variable=self.train_choice, value="Discriminateur").pack(side="left", padx=10)
        
        ttk.Label(frame, text="Fonction de perte :").pack(pady=5)
        self.loss_function = ttk.Combobox(frame, values=["MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss"], width=15)
        self.loss_function.set("MSELoss")
        self.loss_function.pack(pady=5)
        
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
        
        data_frame = ttk.Frame(frame)
        data_frame.pack(pady=10)
        ttk.Label(data_frame, text="Dossier de données :").pack(side="left", padx=5)
        self.data_folder_entry = ttk.Entry(data_frame, textvariable=self.data_folder, width=50)
        self.data_folder_entry.pack(side="left", padx=5)
        browse_btn = ttk.Button(data_frame, text="Parcourir", command=self.select_data_folder)
        browse_btn.pack(side="left", padx=5)
        
        control_frame = ttk.Frame(frame)
        control_frame.pack(pady=10)
        start_btn = ttk.Button(control_frame, text="Démarrer l'Entraînement", command=self.start_training)
        start_btn.pack(side="left", padx=5)
        pause_btn = ttk.Button(control_frame, text="Pause", command=self.pause_training)
        pause_btn.pack(side="left", padx=5)
        resume_btn = ttk.Button(control_frame, text="Reprendre", command=self.resume_training)
        resume_btn.pack(side="left", padx=5)
        switch_btn = ttk.Button(control_frame, text="Switcher", command=self.switch_training)
        switch_btn.pack(side="left", padx=5)
        
        self.training_stats_text = tk.Text(frame, height=8, width=80)
        self.training_stats_text.pack(pady=5)
    
    def select_data_folder(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier de données")
        if folder:
            self.data_folder.set(folder)
    
    def pause_training(self):
        if self.gan_controller:
            self.gan_controller.pause_training()
            self.training_stats_text.insert(tk.END, "Entraînement mis en pause.\n")
    
    def resume_training(self):
        if self.gan_controller:
            self.gan_controller.resume_training()
            self.training_stats_text.insert(tk.END, "Entraînement repris.\n")
    
    def switch_training(self):
        if self.gan_controller:
            self.gan_controller.switch_network()
            self.training_stats_text.insert(tk.END, "Switch effectué : changement du réseau à entraîner.\n")
    
    def start_training(self):
        # Récupération de la configuration du Générateur
        gen_config = {}
        try:
            gen_config["input_size"] = 100  # peut être fixe ou récupérée via un widget
            gen_config["output_size"] = 64  # idem
            gen_config["global_activation"] = self.gen_global_activation.get()
            layers = []
            for (_, layer_type_cb, units_entry, kernel_entry, activation_cb) in self.gen_layer_rows:
                layer = {
                    "layer_type": layer_type_cb.get(),
                    "units": int(units_entry.get()),
                    "kernel_size": int(kernel_entry.get()) if layer_type_cb.get() == "Convolution" and kernel_entry.get() != "" else None,
                    "activation": activation_cb.get()
                }
                layers.append(layer)
            gen_config["layers"] = layers
        except Exception as e:
            messagebox.showerror("Erreur", "Erreur dans la configuration du Générateur: " + str(e))
            return
        
        # Récupération de la configuration du Discriminateur
        disc_config = {}
        try:
            disc_config["input_size"] = 64  # correspond à la sortie du générateur
            disc_config["output_size"] = 1
            disc_config["global_activation"] = self.disc_global_activation.get()
            layers = []
            for (_, layer_type_cb, units_entry, kernel_entry, activation_cb) in self.disc_layer_rows:
                layer = {
                    "layer_type": layer_type_cb.get(),
                    "units": int(units_entry.get()),
                    "kernel_size": int(kernel_entry.get()) if layer_type_cb.get() == "Convolution" and kernel_entry.get() != "" else None,
                    "activation": activation_cb.get()
                }
                layers.append(layer)
            disc_config["layers"] = layers
        except Exception as e:
            messagebox.showerror("Erreur", "Erreur dans la configuration du Discriminateur: " + str(e))
            return
        
        # Configuration d'entraînement
        training_config = {}
        try:
            training_config["learning_rate"] = float(self.learning_rate_entry.get())
            training_config["epochs"] = int(self.epochs_entry.get())
            training_config["batch_size"] = int(self.batch_size_entry.get())
            training_config["data_folder"] = self.data_folder.get()
            training_config["initial_network"] = self.train_choice.get().lower()  # "générateur" ou "discriminateur"
        except Exception as e:
            messagebox.showerror("Erreur", "Erreur dans la configuration d'entraînement: " + str(e))
            return
        
        # Création de l'instance du contrôleur GAN
        self.gan_controller = GANController(gen_config, disc_config, training_config)
        
        # Callback pour afficher les mises à jour durant l'entraînement
        def training_callback(message):
            self.training_stats_text.insert(tk.END, message + "\n")
            self.training_stats_text.see(tk.END)
        
        # Démarrage de l'entraînement dans un thread via le contrôleur
        self.gan_controller.start_training(training_callback)
    
    # ---------------------- Onglet Résumé de la Configuration ----------------------
    def build_summary_tab(self):
        frame = self.summary_frame
        ttk.Label(frame, text="Résumé de la Configuration", font=("Arial", 14)).pack(pady=10)
        self.summary_text = tk.Text(frame, height=18, width=80)
        self.summary_text.pack(pady=10)
        summary_btn = ttk.Button(frame, text="Générer le Résumé", command=self.generate_summary)
        summary_btn.pack(pady=5)
    
    def generate_summary(self):
        summary = "=== Générateur ===\n"
        for (_, layer_type_cb, units_entry, kernel_entry, activation_cb) in self.gen_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            layer_activation = activation_cb.get()
            summary += f"{layer_type}: Units/Filters = {units}, Kernel Size = {kernel}, Activation = {layer_activation}\n"
        summary += f"Activation globale Générateur : {self.gen_global_activation.get()}\n\n"
        summary += "=== Discriminateur ===\n"
        for (_, layer_type_cb, units_entry, kernel_entry, activation_cb) in self.disc_layer_rows:
            layer_type = layer_type_cb.get()
            units = units_entry.get()
            kernel = kernel_entry.get() if layer_type == "Convolution" else "N/A"
            layer_activation = activation_cb.get()
            summary += f"{layer_type}: Units/Filters = {units}, Kernel Size = {kernel}, Activation = {layer_activation}\n"
        summary += f"Activation globale Discriminateur : {self.disc_global_activation.get()}\n\n"
        summary += "=== Paramètres d'Entraînement ===\n"
        summary += f"Réseau à entraîner : {self.train_choice.get()}\n"
        summary += f"Fonction de perte : {self.loss_function.get()}\n"
        summary += f"Learning Rate : {self.learning_rate_entry.get()}\n"
        summary += f"Nombre d'epochs : {self.epochs_entry.get()}\n"
        summary += f"Taille du batch : {self.batch_size_entry.get()}\n"
        summary += f"Dossier de données : {self.data_folder.get()}\n"
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)

if __name__ == "__main__":
    root = tk.Tk()
    app = GanConfigurator(root)
    root.mainloop()
