import yaml
import wandb

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

        return embeddings, labels, original_images
