import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data

@torch.no_grad()
def infer_single_sample(input_sample, causal_encoders, target_assignment, mean, std, device='cpu'):
    # Dictionary to hold model outputs
    causal_outputs = {}
    # Move the input sample to the correct device and prepare it
    input_sample = (input_sample - mean) / std

    input_sample = input_sample.to(device).float()
    # Iterate over each model and its target assignment
    for idx, model in enumerate(causal_encoders):
        # Ensure the model is in evaluation mode
        model.eval()

        # Select the appropriate inputs for this model based on target_assignment
        # Assuming the input sample is already batched or a single sample that mimics a batch of size 1
        model_inputs = input_sample[:, torch.cat([target_assignment[:, 0:1], target_assignment], dim=1).T[idx].bool()].clone().detach()

        # Get the output from the model
        with torch.no_grad():  # Disable gradient computation for inference
            output = model(model_inputs)
            if output.dim() > 1 and output.shape[1] == 1:
                output = output.squeeze(1)  # Correct shape if necessary

        # Store output in dictionary
        causal_outputs[f'model_{idx}'] = output.cpu().numpy()  # Convert to numpy if needed for non-torch downstream processing

    return causal_outputs


class MLP(nn.Module):
    def __init__(self, input_size, output_size, output_type):
        super(MLP, self).__init__()
        self.output_type = output_type
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.layers(x)
        if self.output_type == 'continuous':
            return torch.sigmoid(x)  # Output for continuous variables
        elif self.output_type == 'categorical':
            return x  # Logits for categorical variable

class CausalMappers():
    def __init__(self, causal_encoders, target_assignment, mean, std, device='cpu'):
        self.causal_encoders = [encoder.to(device) for encoder in causal_encoders]
        self.target_assignment = target_assignment.to(device)
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.device = device

    def __call__(self, latents):
        res = infer_single_sample(latents, self.causal_encoders, self.target_assignment, self.mean, self.std, device=self.device)
        
        batch_size = latents.shape[0]
        result_list = [[] for _ in range(batch_size)]
        
        for idx in range(len(self.causal_encoders)):
            model_output = torch.tensor(res[f'model_{idx}'])
            for i in range(batch_size):
                if model_output.dim() > 1 and model_output.shape[1] > 1:
                    result_list[i].append(model_output[i].argmax(-1).item())
                else:
                    result_list[i].append(model_output[i].item())
        
        return result_list

class CausalMapper():
    def __init__(self, causal_mapper, mean, std, target_assignment):
        self.causal_mapper = causal_mapper.eval()
        self.mean = mean.to(causal_mapper.device)
        self.std = std.to(causal_mapper.device)
        self.target_assignment = target_assignment.to(causal_mapper.device)
        self.mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8 : 7} # TODO: make this a parameter
        
    def __call__(self, latents):
        latents = (latents - self.mean) / self.std
        latents = self._prepare_input(latents, self.target_assignment)
        res = self.causal_mapper(latents)
        res = {key: value[i, :].argmax(-1).item() if value.shape[1] == 2 else value[i, :].item() for i, (key, value) in zip(self.mapping.values(), res.items())}
        return list(res.values())
    
    def _prepare_input(self, latents, target_assignment, flatten_inp=True):
        ta = target_assignment.detach()[None,:,:].expand(latents.shape[0], -1, -1)
        latents = torch.cat([latents[:,:,None] * ta, ta], dim=-2).permute(0, 2, 1)
        if flatten_inp:
            latents = latents.flatten(0, 1)
        return latents