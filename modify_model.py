import torch
from core.model import CNNDQN
from core.constants import PRETRAINED_MODELS
from os.path import join

def degrade_model(model_path, noise_factor=0.1):
    # Load the pretrained model structure without specifying output size
    model = CNNDQN((4, 84, 84), 12)  # Match output size to the checkpoint (12 actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Introduce random noise to each parameter
    for param in model.parameters():
        noise = torch.randn_like(param) * noise_factor
        param.data += noise

    # Save the degraded model back to disk
    degraded_model_path = model_path.replace('.dat', '_degraded.dat')
    torch.save(model.state_dict(), degraded_model_path)
    print(f"Degraded model saved at {degraded_model_path}")

if __name__ == '__main__':
    model_file = join(PRETRAINED_MODELS, 'SuperMarioBros-1-1-v0-powerfull.dat')
    degrade_model(model_file, noise_factor=0.2)
