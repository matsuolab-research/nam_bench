import torch

from src import eval_utils


if __name__ == "__main__":
    config_path = "./configs/moving_digits.json" # Please specify your config file. See configs for example.
    output_dir = "./output"
    
    # TODO: Please implement your model here
    model = lambda x: x + 0.1*torch.randn_like(x) # Reconstruction model
    # model = lambda x: torch.randint(0, 10, size=(len(x), )) # Classification model
    
    eval_op = eval_utils.Evaluation(model=model, config_path=config_path)
    eval_op.evaluate(output_dir=output_dir)