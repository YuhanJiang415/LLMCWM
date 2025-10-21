import os
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import argparse
import json
import random

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate causal mappers.')
    parser.add_argument('--environment', type=str, choices=['ithor', 'gridworld'], default='ithor',
                        help='Environment to use (ithor or gridworld).')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='Path to the data folder.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the BISCUITNF model checkpoint.')
    parser.add_argument('--autoencoder_path', type=str, default=None,
                        help='Path to the autoencoder checkpoint.')
    parser.add_argument('--subsample_percentage', type=float, default=0.1,
                        help='Subsample percentage for training dataset.')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Folder to save outputs (models, target assignments, plots).')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (cuda or cpu).')
    args = parser.parse_args()
    return args

def load_model(model_path, autoencoder_path, device='cuda'):
    """
    Load the pre-trained BISCUITNF model.

    Args:
        model_path (str): Path to the BISCUITNF model checkpoint.
        autoencoder_path (str): Path to the autoencoder checkpoint.
        device (str): Device to load the model on.

    Returns:
        model: Loaded BISCUITNF model.
    """
    from models.biscuit_nf import BISCUITNF  # Import here to avoid circular imports
    model = BISCUITNF.load_from_checkpoint(
        model_path, autoencoder_checkpoint=autoencoder_path, map_location=device
    )
    model = model.to(device)
    return model

def load_dataset(environment, data_folder, split='val', subsample_percentage=0.01):
    """
    Load the dataset based on the environment.

    Args:
        environment (str): 'ithor' or 'gridworld'.
        data_folder (str): Path to the data folder.
        split (str): Dataset split to load ('train', 'val', 'test').
        subsample_percentage (float): Percentage of data to subsample.

    Returns:
        dataset: Loaded dataset.
    """
    if environment == 'ithor':
        from experiments.datasets import iTHORDataset
        dataset = iTHORDataset(
            data_folder=data_folder,
            split=split,
            return_targets=False,
            single_image=True,
            return_latents=True,
            triplet=False,
            seq_len=2,
            cluster=False,
            return_text=False,
            subsample_percentage=subsample_percentage
        )
    elif environment == 'gridworld':
        from experiments.datasets import GridworldDataset
        dataset = GridworldDataset(
            data_folder=data_folder,
            split=split,
            return_targets=False,
            single_image=True,
            return_latents=True,
            triplet=False,
            seq_len=2,
            cluster=False,
            return_text=False,
            subsample_percentage=subsample_percentage
        )
    else:
        raise ValueError(f"Unknown environment: {environment}")
    return dataset

def encode_dataset(model, dataset, batch_size=32, device='cpu'):
# def encode_dataset(model, dataset, batch_size=256, device='cpu'):
    """
    Encode the dataset using the model's encoder.

    Args:
        model: The pre-trained model with an encoder.
        dataset: The dataset to encode.
        batch_size (int): Batch size for encoding.
        device (str): Device to perform encoding on.

    Returns:
        full_dataset: TensorDataset containing encoded inputs and latents.
    """
    loader = data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=False
    )
    all_encs, all_latents = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding dataset"):
            inps, *_, latents = batch
            inps = model.autoencoder.encoder(inps.to(device))
            encs = model.encode(inps)
            all_encs.append(encs.cpu())
            all_latents.append(latents)

    all_encs = torch.cat(all_encs, dim=0)
    all_latents = torch.cat(all_latents, dim=0)

    full_dataset = data.TensorDataset(all_encs, all_latents)
    return full_dataset

def train_causal_encoder(model, dataset, device='cuda'):
    """
    Train the causal encoder to predict causal factors from latent variables.

    Args:
        model: The pre-trained model.
        dataset: The encoded dataset.
        device (str): Device to perform training on.

    Returns:
        r2_matrix: R^2 matrix of the trained model.
        encoder: Trained causal encoder.
        target_assignment: Target assignment matrix.
        all_encs_mean: Mean of the encodings.
        all_encs_std: Standard deviation of the encodings.
    """
    # Prepare data
    all_encs, all_latents = dataset.tensors
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(
        dataset, lengths=[train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Set target assignment
    if hasattr(model, 'target_assignment') and model.target_assignment is not None:
        target_assignment = model.target_assignment.clone().to(device)
    else:
        target_assignment = torch.eye(all_encs.shape[-1]).to(device)

    # Train the causal encoder
    encoder, losses = train_network(model, train_dataset, target_assignment, device)

    # Evaluate the model and compute R^2 matrix
    test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
    test_inps = test_inps.to(device)
    test_labels = test_labels.to(device)
    test_exp_inps, test_exp_labels = prepare_input(
        test_inps, target_assignment, test_labels, flatten_inp=False
    )
    pred_dict = encoder.forward(test_exp_inps.to(device))
    _, _, norm_dists = encoder.calculate_loss_distance(pred_dict, test_exp_labels)

    avg_norm_dists, r2_matrix = log_r2_statistic(
        encoder, pred_dict, test_labels, norm_dists
    )

    return r2_matrix, encoder, target_assignment, losses

# def prepare_input(inps, target_assignment, latents, flatten_inp=True):
#     # Move inputs and latents to the device of target_assignment
#     device = target_assignment.device
#     inps = inps.to(device)
#     latents = latents.to(device)
    
#     ta = target_assignment.detach()[None, :, :].expand(inps.shape[0], -1, -1)
#     inps = torch.cat([inps[:, :, None] * ta, ta], dim=-2).permute(0, 2, 1)
#     latents = latents[:, None].expand(-1, inps.shape[1], -1)
#     if flatten_inp:
#         inps = inps.flatten(0, 1)
#         latents = latents.flatten(0, 1)
#     return inps, latents

def prepare_input(inps_latent, target_assignment, latents_gt, flatten_inp=True):
    """
    [MODIFIED FUNCTION]
    Prepares inputs (inps) and labels (latents) for the CausalEncoder.
    Ensures variable scopes are clean to prevent dimension mismatch.
    """
    device = target_assignment.device

    # 1. Move original data to device
    inps = inps_latent.to(device)     # Shape (B, 40)
    latents = latents_gt.to(device)    # Shape (B, 40)

    # 2. Get original shape dimensions (B, 40)
    orig_batch_size = inps.shape[0]
    orig_latent_dim = inps.shape[1]  # This is 40

    # 3. Create the expanded input features for the model
    ta = target_assignment.detach()[None, :, :].expand(orig_batch_size, -1, -1) # Shape (B, 40, 40)

    # 'inps' (B, 40, 1) * 'ta' (B, 40, 40) -> (B, 40, 40)
    inps_features_part1 = inps[:, :, None] * ta

    # Concat part1 (B, 40, 40) and 'ta' (B, 40, 40) along dim=-2 (the '40' dim)
    inps_cat = torch.cat([inps_features_part1, ta], dim=-2) # Shape (B, 80, 40)

    # Permute to (B, 40, 80) so model can iterate over 40 latents, each with 80 features
    inps_expanded = inps_cat.permute(0, 2, 1) # Shape (B, 40, 80)

    # 4. Create the expanded ground truth labels
    # We must use the *original* latent dim (40), not the new feature dim (80)
    latents_expanded = latents[:, None].expand(-1, orig_latent_dim, -1) # Shape (B, 40, NumCausalVars)

    # 5. Flatten if needed
    if flatten_inp:
        inps_expanded = inps_expanded.flatten(0, 1)
        latents_expanded = latents_expanded.flatten(0, 1)

    return inps_expanded, latents_expanded

def train_network(model, train_dataset, target_assignment, device, num_epochs=100):
    """
    Train the causal encoder network.

    Args:
        model: The pre-trained model.
        train_dataset: The training dataset.
        target_assignment: Target assignment matrix.
        device (str): Device to perform training on.
        num_epochs (int): Number of training epochs.

    Returns:
        encoder: Trained causal encoder.
        losses: Training losses over epochs.
    """
    from models.shared import CausalEncoder
    causal_var_info = model.hparams.causal_var_info
    encoder = CausalEncoder(
        c_hid=256,
        lr=3e-3,
        causal_var_info=causal_var_info,
        single_linear=True,
        c_in=model.hparams.num_latents * 2,
        warmup=0,
    ).float().to(device)

    optimizer = optim.Adam(encoder.parameters(), lr=3e-3)
    # train_loader = data.DataLoader(
    #     train_dataset, batch_size=512, shuffle=True, drop_last=False
    # )
    train_loader = data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=False
    )
    target_assignment = target_assignment.to(device)
    encoder.train()
    losses = []

    for epoch in tqdm(range(num_epochs), desc="Training causal encoder"):
        avg_loss = 0.0
        for inps, latents in train_loader:
            inps = inps.to(device).float()
            latents = latents.to(device).float()
            inps, latents = prepare_input(inps, target_assignment, latents)
            loss = encoder._get_loss([inps, latents], mode=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        losses.append(avg_loss / len(train_loader))

    return encoder, losses

def log_r2_statistic(encoder, pred_dict, test_labels, norm_dists):
    """
    Calculate R^2 statistics for the predictions.

    Args:
        encoder: The trained causal encoder.
        pred_dict: Dictionary of model predictions.
        test_labels: Ground truth labels.
        norm_dists: Normalized distances.

    Returns:
        avg_norm_dists: Average normalized distances.
        r2_matrix: R^2 matrix.
    """
    causal_var_info = encoder.hparams.causal_var_info
    avg_pred_dict = OrderedDict()
    for i, var_key in enumerate(causal_var_info):
        var_info = causal_var_info[var_key]
        gt_vals = test_labels[..., i]
        if var_info.startswith('continuous'):
            avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0],)
        elif var_info.startswith('angle'):
            avg_angle = torch.atan2(
                torch.sin(gt_vals).mean(dim=0, keepdim=True),
                torch.cos(gt_vals).mean(dim=0, keepdim=True),
            ).expand(gt_vals.shape[0],)
            avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2 * np.pi, avg_angle)
            avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
        elif var_info.startswith('categ'):
            gt_vals = gt_vals.long()
            mode = torch.mode(gt_vals, dim=0, keepdim=True).values
            avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0], -1)
        else:
            raise ValueError(f'Unknown variable type: "{var_key}" in R^2 statistics.')

    _, _, avg_norm_dists = encoder.calculate_loss_distance(
        avg_pred_dict, test_labels, keep_sign=True
    )

    r2_matrix = []
    for var_key in causal_var_info:
        ss_res = (norm_dists[var_key] ** 2).mean(dim=0)
        ss_tot = (avg_norm_dists[var_key] ** 2).mean(dim=0, keepdim=True)
        ss_tot = torch.where(ss_tot == 0.0, torch.tensor(1.0), ss_tot)
        r2 = 1 - ss_res / ss_tot
        r2_matrix.append(r2)
    r2_matrix = torch.stack(r2_matrix, dim=-1).cpu().detach().numpy()
    return avg_norm_dists, r2_matrix

def construct_target_assignment(r2_matrix, threshold=0.1, environment='ithor'):
    """
    Construct the target assignment matrix based on the R^2 matrix.

    Args:
        r2_matrix: The R^2 matrix.
        threshold (float): Threshold for assigning targets.
        environment (str): The environment ('ithor' or 'gridworld').

    Returns:
        ta: Target assignment matrix.
    """
    r2_matrix = torch.from_numpy(r2_matrix)
    r2_matrix_ = r2_matrix / r2_matrix.abs().max(dim=0, keepdims=True).values.clamp(min=threshold)
    max_values, max_indices = r2_matrix_.max(dim=1, keepdim=True)
    mask = max_values > threshold
    ta = torch.zeros_like(r2_matrix_, dtype=torch.bool)
    ta.scatter_(1, max_indices, mask)

    if environment == 'ithor':
        ta = torch.cat([
            ta[:, :1],
            ta[:, 1:7].sum(dim=-1, keepdims=True),
            ta[:, 7:9],
            ta[:, 9:13].sum(dim=-1, keepdims=True),
            ta[:, 13:],
        ], dim=-1)
    elif environment == 'gridworld':
        ta = torch.cat([
            ta[:, :2].sum(dim=-1, keepdims=True),
            ta[:, 2:4],
            ta[:, 4:6],
            ta[:, 6:7],
            ta[:, 7:8],
        ], dim=-1)
    else:
        raise ValueError(f"Unknown environment: {environment}")

    return ta

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for causal variable prediction.
    """
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
            return torch.sigmoid(x)
        elif self.output_type == 'categorical':
            return x

class MultiHeadMLP(nn.Module):
    """
    Multi-Head MLP model for baseline causal variable prediction.
    """
    def __init__(self, input_size, output_sizes, output_types):
        super(MultiHeadMLP, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleList()
        self.output_types = output_types
        for output_size, output_type in zip(output_sizes, output_types):
            if output_type == 'continuous':
                head = nn.Sequential(
                    nn.Linear(128, output_size),
                    nn.Sigmoid()
                )
            else:
                head = nn.Linear(128, output_size)
            self.heads.append(head)

    def forward(self, x):
        x = self.shared_layers(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs

def train_causal_mappers(causal_encoders, optimizers, train_loader, val_loader, ta_expanded, device='cuda', epochs=50):
    """
    Train the causal mappers (MLPs) for each causal variable.
    """
    model_losses = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        # Training phase
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            for idx, (model, optimizer) in enumerate(zip(causal_encoders, optimizers)):
                optimizer.zero_grad()
                model_inputs = inputs[:, ta_expanded.T[idx].bool()].float().clone().detach()
                model_targets = targets[:, idx].float().clone().detach()

                if 'categorical' in model.output_type:
                    model_targets = model_targets.long()

                outputs = model(model_inputs)
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)

                if model.output_type == 'continuous':
                    loss = F.mse_loss(outputs, model_targets)
                else:
                    loss = F.cross_entropy(outputs, model_targets)

                loss.backward()
                optimizer.step()

        # Validation phase
        val_losses = []
        with torch.no_grad():
            for idx, model in enumerate(causal_encoders):
                val_loss = 0.0
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_model_inputs = val_inputs[:, ta_expanded.T[idx].bool()].float().clone().detach()
                    val_model_targets = val_targets[:, idx].float().clone().detach()

                    if 'categorical' in model.output_type:
                        val_model_targets = val_model_targets.long()

                    val_outputs = model(val_model_inputs)
                    if val_outputs.dim() > 1 and val_outputs.shape[1] == 1:
                        val_outputs = val_outputs.squeeze(1)

                    if model.output_type == 'continuous':
                        val_loss += F.mse_loss(val_outputs, val_model_targets).item()
                    else:
                        val_loss += F.cross_entropy(val_outputs, val_model_targets).item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f'Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}')

    return causal_encoders

def evaluate_causal_mappers(causal_encoders, test_dataset, ta_expanded, causal_var_info, output_folder, environment, device='cuda'):
    """
    Evaluate the causal mappers on the test dataset.
    """
    # test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    all_predictions = []
    all_ground_truths = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs, targets = batch
        inputs = inputs.to(device)

        batch_predictions = []
        for idx, model in enumerate(causal_encoders):
            model.eval()
            model_inputs = inputs[:, ta_expanded.T[idx].bool()].float().clone().detach()
            with torch.no_grad():
                outputs = model(model_inputs)
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                if 'categorical' in model.output_type:
                    outputs = outputs.argmax(dim=-1)
                batch_predictions.append(outputs.cpu().numpy())

        all_predictions.append(np.column_stack(batch_predictions))
        all_ground_truths.append(targets.cpu().numpy())

    # Convert lists to arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)

    # Calculate MAE
    mae = np.mean(np.abs(all_predictions - all_ground_truths), axis=0)
    dimensions = np.arange(len(mae))

    # Plot MAE
    plt.figure(figsize=(10, 5))
    plt.bar(dimensions, mae, tick_label=dimensions)
    plt.xlabel('Dimension')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'MAE for Each Dimension - {environment}')
    plt.xticks(dimensions, causal_var_info.keys(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'causal_mappers_mae_{environment}.png'))
    plt.close()

def train_multi_head_model(model, train_loader, val_loader, causal_var_info, device='cuda', epochs=50):
    """
    Train the multi-head MLP baseline model.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model_losses = {'train_loss': [], 'val_loss': []}

    keys = list(causal_var_info.keys())
    output_types = ['continuous' if 'continuous' in causal_var_info[key] else 'categorical' for key in keys]

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Training)', leave=False)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = 0
            for idx, output in enumerate(outputs):
                target = targets[:, idx]
                if output_types[idx] == 'continuous':
                    loss += F.mse_loss(output.squeeze(), target)
                else:
                    loss += F.cross_entropy(output, target.long())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix(train_loss=loss.item())
        train_pbar.close()

        train_loss /= len(train_loader)
        model_losses['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Validation)', leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = 0
                for idx, output in enumerate(outputs):
                    target = targets[:, idx]
                    if output_types[idx] == 'continuous':
                        loss += F.mse_loss(output.squeeze(), target)
                    else:
                        loss += F.cross_entropy(output, target.long())
                val_loss += loss.item()
                val_pbar.set_postfix(val_loss=loss.item())
        val_pbar.close()

        val_loss /= len(val_loader)
        model_losses['val_loss'].append(val_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return model_losses

def evaluate_baseline_model(model, dataset, causal_var_info, output_folder, environment, device='cuda'):
    """
    Evaluate the baseline model on the test dataset.
    """
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_ground_truths = []
    # dataloader = data.DataLoader(dataset, batch_size=512, shuffle=False)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=False)

    keys = list(causal_var_info.keys())
    output_types = ['continuous' if 'continuous' in causal_var_info[key] else 'categorical' for key in keys]

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating Baseline Model"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            batch_predictions = []
            for idx, output in enumerate(outputs):
                if output_types[idx] == 'continuous':
                    batch_predictions.append(output.squeeze().cpu().numpy())
                else:
                    batch_predictions.append(output.argmax(dim=-1).cpu().numpy())

            all_predictions.append(np.column_stack(batch_predictions))
            all_ground_truths.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)

    # Calculate MAE
    mae = np.mean(np.abs(all_predictions - all_ground_truths), axis=0)

    # Plot MAE
    dimensions = np.arange(len(mae))
    plt.figure(figsize=(10, 5))
    plt.bar(dimensions, mae, tick_label=dimensions)
    plt.xlabel('Dimension')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'MAE for Each Dimension - Baseline Model ({environment})')
    plt.xticks(dimensions, causal_var_info.keys(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'baseline_model_mae_{environment}.png'))
    plt.close()


def infer_single_sample(latents, causal_encoders, target_assignment, mean, std, device='cpu'):
    """
    Perform inference on a single input sample using trained causal encoders.

    Args:
        latents: The latent representation (batched) to use as input.
        causal_encoders: List of trained causal encoders.
        target_assignment: Target assignment matrix.
        mean: Mean used for normalization.
        std: Standard deviation used for normalization.
        device: Device to perform inference on.

    Returns:
        causal_outputs: Dictionary containing predictions for each encoder.
    """
    latents = (latents - mean) / std
    latents = latents.to(device).float()

    causal_outputs = {}

    for idx, model in enumerate(causal_encoders):
        model.eval()


        model_inputs = latents[:, target_assignment.T[idx].bool()].clone().detach()

        with torch.no_grad():
            output = model(model_inputs)

        if output.dim() == 2 and output.shape[1] == 1:
            output = output.squeeze(1)

        causal_outputs[f'model_{idx}'] = output.cpu().numpy()

    return causal_outputs



class CausalMappers:
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

def main():
    set_seed()

    args = parse_arguments()
    environment = args.environment

    if args.data_folder is None:
        if environment == 'ithor':
            data_folder = 'data/ithor/'
        elif environment == 'gridworld':
            data_folder = 'data/gridworld/'
    else:
        data_folder = args.data_folder

    if args.model_path is None:
        if environment == 'ithor':
            model_path = 'pretrained_models/ithor_biscuit.ckpt'
        elif environment == 'gridworld':
            model_path = 'pretrained_models/gridworld_biscuit.ckpt'
    else:
        model_path = args.model_path

    if args.autoencoder_path is None:
        if environment == 'ithor':
            autoencoder_path = 'pretrained_models/AE_iTHOR/AE_40l_64hid.ckpt'
        elif environment == 'gridworld':
            autoencoder_path = 'pretrained_models/AE_gridworld/AE_40l_64hid.ckpt'
    else:
        autoencoder_path = args.autoencoder_path

    if args.output_folder is None:
        output_folder = f'causal_mappers_outputs_{environment}/'
    else:
        output_folder = args.output_folder

    subsample_percentage = args.subsample_percentage
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(output_folder, exist_ok=True)

    model = load_model(model_path, autoencoder_path, device)

    # -------------------- Training and Validation Data Preparation --------------------

    subsample_str = f"subsample_{subsample_percentage}"
    encoded_train_dataset_path = os.path.join(output_folder, f'encoded_train_dataset_{subsample_str}_{environment}.pt')
    normalization_stats_path = os.path.join(output_folder, f'normalization_stats_{subsample_str}_{environment}.pt')

    if os.path.exists(encoded_train_dataset_path) and os.path.exists(normalization_stats_path):
        print(f"Loading encoded train dataset from {encoded_train_dataset_path}")
        encoded_dataset = torch.load(encoded_train_dataset_path)
        normalization_stats = torch.load(normalization_stats_path)
        all_encs_mean = normalization_stats['mean']
        all_encs_std = normalization_stats['std']
    else:
        train_dataset_raw = load_dataset(environment, data_folder, split='val', subsample_percentage=subsample_percentage)
        encoded_dataset = encode_dataset(
            model,
            train_dataset_raw,
            device=device
        )
        torch.save(encoded_dataset, encoded_train_dataset_path)

        train_encs, train_latents = encoded_dataset.tensors
        all_encs_mean = train_encs.mean(dim=0, keepdim=True)
        all_encs_std = train_encs.std(dim=0, keepdim=True).clamp(min=1e-2)

        torch.save({'mean': all_encs_mean, 'std': all_encs_std}, normalization_stats_path)

    train_encs, train_latents = encoded_dataset.tensors
    train_encs = (train_encs - all_encs_mean) / all_encs_std
    encoded_dataset = data.TensorDataset(train_encs, train_latents)

    train_size = int(0.8 * len(encoded_dataset))
    val_size = len(encoded_dataset) - train_size
    train_dataset, val_dataset = data.random_split(
        encoded_dataset, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # train_loader = data.DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=False)
    # val_loader = data.DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

    # -------------------- Train Causal Encoder and Compute R^2 Matrix --------------------

    r2_matrix, encoder, target_assignment, losses = train_causal_encoder(model, encoded_dataset, device)

    np.save(os.path.join(output_folder, f'r2_matrix_{environment}.npy'), r2_matrix)
    torch.save(target_assignment, os.path.join(output_folder, f'target_assignment_{environment}.pt'))

    ta = construct_target_assignment(r2_matrix, environment=environment)

    # -------------------- Prepare Causal Mappers --------------------

    causal_var_info = model.hparams.causal_var_info
    if environment == 'ithor':
        ta_expanded = torch.cat([
            ta[:, :1].repeat(1, 1),
            ta[:, 1:2].repeat(1, 6),
            ta[:, 2:4].repeat(1, 1),
            ta[:, 4:5].repeat(1, 4),
            ta[:, 5:].repeat(1, 1)
        ], dim=1)
    elif environment == 'gridworld':
        ta_expanded = torch.cat([ta[:, 0:1], ta], dim=1)
    else:
        raise ValueError(f"Unknown environment: {environment}")

    causal_encoders = []
    optimizers = []
    for idx, (key, value) in enumerate(causal_var_info.items()):
        if environment == 'ithor':
            output_size = int(value.split('_')[1][0])
        else:
            output_size = int(value.split('_')[1])
        output_type = 'continuous' if 'continuous' in value else 'categorical'
        input_size = int(ta_expanded[:, idx].sum().item())
        causal_enc = MLP(input_size=input_size, output_size=output_size, output_type=output_type)
        causal_encoders.append(causal_enc.to(device))
        optimizers.append(optim.Adam(causal_enc.parameters(), lr=1e-3, weight_decay=1e-5))

    # -------------------- Train Causal Mappers --------------------

    causal_encoders = train_causal_mappers(causal_encoders, optimizers, train_loader, val_loader, ta_expanded, device)

    # -------------------- Test Data Preparation --------------------

    encoded_test_dataset_path = os.path.join(output_folder, f'encoded_test_dataset_{environment}.pt')
    if os.path.exists(encoded_test_dataset_path):
        print(f"Loading encoded test dataset from {encoded_test_dataset_path}")
        encoded_test_dataset = torch.load(encoded_test_dataset_path)
    else:
        test_dataset_raw = load_dataset(environment, data_folder, split='test', subsample_percentage=1.0)
        encoded_test_dataset = encode_dataset(
            model,
            test_dataset_raw,
            device=device
        )
        torch.save(encoded_test_dataset, encoded_test_dataset_path)

    test_encs, test_latents = encoded_test_dataset.tensors
    test_encs = (test_encs - all_encs_mean) / all_encs_std
    encoded_test_dataset = data.TensorDataset(test_encs, test_latents)

    # -------------------- Evaluate Causal Mappers on Test Data --------------------

    evaluate_causal_mappers(causal_encoders, encoded_test_dataset, ta_expanded, causal_var_info, output_folder, environment, device)

    # -------------------- Save Additional Outputs --------------------

    print(ta_expanded.shape)
    causal_mappers = CausalMappers(causal_encoders, ta_expanded, all_encs_mean, all_encs_std, device=device)

    # Perform inference on a sample batch from the training data to test the mappers
    inp, _ = next(iter(train_loader))
    res = causal_mappers(inp.to(device))

    to_save = {
        'causal_encoders': causal_encoders,
        'target_assignment': ta,
        'all_encs_mean': all_encs_mean,
        'all_encs_std': all_encs_std,
        'causal_var_info': causal_var_info
    }
    save_path = os.path.join(output_folder, f'causal_encoders_{environment}.pt')
    torch.save(to_save, save_path)
    print(f"Causal mappers saved to {save_path}")

    # -------------------- Baseline Model Training and Evaluation --------------------

    input_size = train_encs.shape[1]
    if environment == 'ithor':
        output_sizes = [int(value.split('_')[1][0]) for value in causal_var_info.values()]
    else:
        output_sizes = [int(value.split('_')[1]) for value in causal_var_info.values()]
    output_types = ['continuous' if 'continuous' in value else 'categorical' for value in causal_var_info.values()]

    multi_head_model = MultiHeadMLP(input_size=input_size, output_sizes=output_sizes, output_types=output_types)
    train_multi_head_model(multi_head_model, train_loader, val_loader, causal_var_info, device=device)

    torch.save(multi_head_model.state_dict(), os.path.join(output_folder, f'baseline_model_{environment}.pt'))

    evaluate_baseline_model(multi_head_model, encoded_test_dataset, causal_var_info, output_folder, environment, device=device)

    # -------------------- Save Additional Outputs --------------------

    with open(os.path.join(output_folder, f'causal_var_info_{environment}.json'), 'w') as f:
        json.dump(causal_var_info, f)

if __name__ == '__main__':
    main()
