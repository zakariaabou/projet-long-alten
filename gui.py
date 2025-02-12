import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Configuration d'un Réseau GAN")
    gr.Markdown("Configurez l'architecture du générateur et du discriminateur, contrôlez l'entraînement et visualisez un résumé de la configuration.")

    with gr.Tabs():
        ## Onglet "Générateur"
        with gr.Tab("Générateur"):
            gr.Markdown("## Configuration du Générateur")
            # Dataframe pour ajouter plusieurs couches dynamiquement.
            gen_layers_df = gr.Dataframe(
                headers=["Layer Type", "Units/Filters", "Kernel Size"],
                datatype=["str", "number", "number"],
                row_count=(1, "dynamic"),
                label="Couches du Générateur"
            )
            gen_activation = gr.Dropdown(
                label="Fonction d'activation",
                choices=["relu", "tanh", "sigmoid"],
                value="relu"
            )
            gen_preview_btn = gr.Button("Prévisualiser Générateur")
            gen_summary = gr.Textbox(label="Résumé du Générateur", interactive=False, lines=5)

            def preview_generator(layers, activation):
                summary = "Configuration du Générateur :\n"
                if layers is None or len(layers) == 0:
                    summary += "Aucune couche configurée.\n"
                else:
                    for row in layers:
                        # Each row: [Layer Type, Units/Filters, Kernel Size]
                        layer_type = row[0] if row[0] else "Undefined"
                        units = row[1] if row[1] is not None else "Undefined"
                        kernel = row[2] if row[2] is not None else "N/A"
                        summary += f"- {layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
                summary += f"Activation : {activation}"
                return summary

            gen_preview_btn.click(
                fn=preview_generator,
                inputs=[gen_layers_df, gen_activation],
                outputs=gen_summary
            )

        ## Onglet "Discriminateur"
        with gr.Tab("Discriminateur"):
            gr.Markdown("## Configuration du Discriminateur")
            disc_layers_df = gr.Dataframe(
                headers=["Layer Type", "Units/Filters", "Kernel Size"],
                datatype=["str", "number", "number"],
                row_count=(1, "dynamic"),
                label="Couches du Discriminateur"
            )
            disc_activation = gr.Dropdown(
                label="Fonction d'activation",
                choices=["relu", "tanh", "sigmoid"],
                value="relu"
            )
            disc_preview_btn = gr.Button("Prévisualiser Discriminateur")
            disc_summary = gr.Textbox(label="Résumé du Discriminateur", interactive=False, lines=5)

            def preview_discriminator(layers, activation):
                summary = "Configuration du Discriminateur :\n"
                if layers is None or len(layers) == 0:
                    summary += "Aucune couche configurée.\n"
                else:
                    for row in layers:
                        layer_type = row[0] if row[0] else "Undefined"
                        units = row[1] if row[1] is not None else "Undefined"
                        kernel = row[2] if row[2] is not None else "N/A"
                        summary += f"- {layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
                summary += f"Activation : {activation}"
                return summary

            disc_preview_btn.click(
                fn=preview_discriminator,
                inputs=[disc_layers_df, disc_activation],
                outputs=disc_summary
            )

        ## Onglet "Contrôle d'Entraînement"
        with gr.Tab("Contrôle d'Entraînement"):
            gr.Markdown("## Contrôle de l'Entraînement")
            train_choice = gr.Radio(
                label="Réseau à entraîner",
                choices=["Générateur", "Discriminateur"],
                value="Générateur"
            )
            pause_btn = gr.Button("Pause Entraînement")
            resume_btn = gr.Button("Reprendre Entraînement")
            switch_btn = gr.Button("Switcher Réseau")
            training_stats = gr.Textbox(label="Statistiques d'Entraînement", interactive=False, lines=5)

            def pause_training():
                return "Entraînement mis en pause."

            def resume_training():
                return "Entraînement repris."

            def switch_training():
                return "Switch effectué : changement du réseau à entraîner."

            pause_btn.click(fn=pause_training, inputs=[], outputs=training_stats)
            resume_btn.click(fn=resume_training, inputs=[], outputs=training_stats)
            switch_btn.click(fn=switch_training, inputs=[], outputs=training_stats)

        ## Onglet "Résumé de la Configuration"
        with gr.Tab("Résumé de la Configuration"):
            gr.Markdown("## Récapitulatif de la Configuration")
            config_summary = gr.Textbox(label="Résumé complet", interactive=False, lines=10)
            summary_btn = gr.Button("Générer le Résumé")

            def generate_summary(gen_layers, gen_activation, disc_layers, disc_activation, train_choice):
                summary = "--- Générateur ---\n"
                if gen_layers is None or len(gen_layers) == 0:
                    summary += "Aucune couche configurée pour le Générateur.\n"
                else:
                    for row in gen_layers:
                        layer_type = row[0] if row[0] else "Undefined"
                        units = row[1] if row[1] is not None else "Undefined"
                        kernel = row[2] if row[2] is not None else "N/A"
                        summary += f"{layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
                summary += f"Activation Générateur : {gen_activation}\n\n"
                summary += "--- Discriminateur ---\n"
                if disc_layers is None or len(disc_layers) == 0:
                    summary += "Aucune couche configurée pour le Discriminateur.\n"
                else:
                    for row in disc_layers:
                        layer_type = row[0] if row[0] else "Undefined"
                        units = row[1] if row[1] is not None else "Undefined"
                        kernel = row[2] if row[2] is not None else "N/A"
                        summary += f"{layer_type}: Units/Filters = {units}, Kernel Size = {kernel}\n"
                summary += f"Activation Discriminateur : {disc_activation}\n\n"
                summary += f"Réseau à entraîner : {train_choice}"
                return summary

            summary_btn.click(
                fn=generate_summary,
                inputs=[gen_layers_df, gen_activation, disc_layers_df, disc_activation, train_choice],
                outputs=config_summary
            )

demo.launch()
