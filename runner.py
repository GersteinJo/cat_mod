import yaml
from metric_modules import encode_dataset, train_classifier, compare_embeddings

class Runner:
    def __init__(self, 
                 config_path,
                 encoder_function,
                 loader,
                 original_loader, # in case encoder needs somewhat modified version of original image to proceed
                 cat_mod,
                ):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.encoder_function = encoder_function
        self.loader = loader
        self.original_loader = original_loader
        self.cat_mod = cat_mod
    
    def run():
        embeddings, labels, original_images = encode_dataset(self.encoder_function, self.loader, self.original_loader)
        
        run_name = f'encoder_{config["encoder"]}_kernel_{config["kernel"]}_num_exemplars_{config["num_exemp"]}'
        logger = wandb.init(project = 'encoder_runs', config = config, name = run_name,
            tags = [config['model_name'], config['encoder'], config['kernel'], f"num_exemp_{config['num_exemp']}"])
        train_classifier(config, encoded_data=embeddings, labels=labels, logger = logger)
        logger.log({"second_order_similarity" : compare_embeddings(embeddings, original_images)})
        logger.finish()
        
        
        
            
