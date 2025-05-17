import yaml
import wandb
from metric_modules import train_classifier, encode_dataset, compare_embeddings, embedding_plotter

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

class Runner:
    def __init__(self,
                 encoder_function,
                 loader,
                 original_loader, # in case encoder needs somewhat modified version of original image to proceed
                 config_path = None,
                ):
        if config_path:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                print(self.config)
        else:
            self.config = None
        self.encoder_function = encoder_function
        self.loader = loader
        self.original_loader = original_loader

    def run(self, run_name = None, device = "cpu", n_iter = 1, embeddings=None, labels=None, original_images=None):
        if embeddings is None or labels is None or original_images is None:
            embeddings, labels, original_images = encode_dataset(self.encoder_function, self.loader, self.original_loader, device)

        for _ in range(n_iter):
            permutation = np.arange(len(labels))
            np.random.shuffle(permutation)
            if run_name is None : run_name = f'encoder_{self.config["encoder"]}_kernel_{self.config["kernel"]}_num_exemplars_{self.config["num_exemp"]}'
            logger = wandb.init(project = 'encoder_runs', config = self.config, name = run_name,
                tags = [self.config['model_name'], self.config['encoder'], self.config['kernel'], f"num_exemp_{self.config['num_exemp']}"]) if self.config else None

            train_classifier(self.config, encoded_data=embeddings[permutation], labels=labels[permutation], logger = logger)

            logger.log({"second_order_similarity" : compare_embeddings(embeddings[permutation[:10000]], original_images[permutation[:10000]])})

            logger.finish()

        tsne = TSNE(n_components=2, random_state=1,
                    init='pca', n_iter=5000,
                    metric='euclidean')

        # Fit and transform your data
        tsne_results = tsne.fit_transform(embeddings[:1000])

        # Prepare the data DataFrame correctly
        data_df = pd.DataFrame({
            'label': np.array(labels[:1000])  # Assuming you have labels
            # Add any other columns you want for hover information
        })

        data_df['tsne_x'] = tsne_results[:,0]
        data_df['tsne_y'] = tsne_results[:,1]
        # Call the plotting function correctly
        embedding_plotter(
            embedding=tsne_results,  # This should be your 2D t-SNE results (1000x2 array)
            data=data_df,            # This contains your labels and other metadata
            hue='label',             # Column name in data_df to use for coloring
        )

        return embeddings, labels, original_images, data_df
