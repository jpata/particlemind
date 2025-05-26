#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from geomloss import SamplesLoss
from geomloss.kernel_samples import gaussian_kernel as geomloss_gaussian_kernel
from scipy.optimize import linear_sum_assignment

from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import io
import PIL.Image
import torch.profiler
import yaml
import argparse

import torch_cluster
import torch_scatter
TORCH_GEOMETRIC_STACK_AVAILABLE = True
print("torch_cluster and torch_scatter found. VoxelTransformerClusterer will be available.")

CUML_AVAILABLE = True
from cuml.cluster import DBSCAN as cuMLDBSCAN



# In[2]:


def fig_to_pil(fig):
    """Converts a Matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = PIL.Image.open(buf).convert("RGB") # Ensure RGB
    return image

def generate_pseudo_data(
    max_true_clusters: int,
    max_hits_per_batch_item: int,
    avg_true_clusters: float,
    avg_hits_per_cluster: float,
    batch_size: int,
    cluster_pos_feature_dim: int = 3,
    hit_pos_feature_dim: int = 3,
    ensure_min_one_hit_per_cluster: bool = False
):
    """
    Generates randomized pseudo-data for hits and their true cluster centers.
    Includes assertions to check internal consistency of generated cluster counts.
    """
    if cluster_pos_feature_dim <= 0 :
        cluster_pos_feature_dim = 3
    if hit_pos_feature_dim <= 0:
        hit_pos_feature_dim = 3

    all_H_padded_list = []
    all_C_padded_list = []
    all_true_k_list = []
    all_true_num_hits_list = []

    for item_idx in range(batch_size):
        mu_k = avg_true_clusters
        sigma_k = 0.1 * avg_true_clusters if avg_true_clusters > 0 else 0.1
        num_true_clusters_sample = np.random.normal(loc=mu_k, scale=sigma_k)
        num_true_clusters = max(0, int(round(num_true_clusters_sample)))
        num_true_clusters = min(num_true_clusters, max_true_clusters)
        all_true_k_list.append(num_true_clusters)

        current_C_padded_np = np.zeros((max_true_clusters, cluster_pos_feature_dim), dtype=np.float32)
        generated_hits_for_batch_item = []

        if num_true_clusters > 0:
            true_cluster_positions_np = np.random.uniform(-1, 1, size=(num_true_clusters, cluster_pos_feature_dim)).astype(np.float32)
            current_C_padded_np[:num_true_clusters, :] = true_cluster_positions_np

            for i in range(num_true_clusters):
                true_cluster_pos = true_cluster_positions_np[i]
                mu_h_c = avg_hits_per_cluster
                sigma_h_c = 0.1 * avg_hits_per_cluster if avg_hits_per_cluster > 0 else 0.1
                num_hits_for_this_cluster_sample = np.random.normal(loc=mu_h_c, scale=sigma_h_c)

                if ensure_min_one_hit_per_cluster:
                    num_hits_for_this_cluster = max(1, int(round(num_hits_for_this_cluster_sample)))
                else:
                    num_hits_for_this_cluster = max(0, int(round(num_hits_for_this_cluster_sample)))

                if num_hits_for_this_cluster > 0:
                    hit_displacements = np.random.normal(loc=0.0, scale=0.02, size=(num_hits_for_this_cluster, hit_pos_feature_dim))
                    hits_for_this_cluster = true_cluster_pos[:hit_pos_feature_dim] + hit_displacements
                    hits_for_this_cluster = np.clip(hits_for_this_cluster, -1, 1).astype(np.float32)
                    generated_hits_for_batch_item.extend(list(hits_for_this_cluster))

        if num_true_clusters > 0:
            assert np.any(current_C_padded_np[num_true_clusters-1, :] != 0), \
                f"Assertion Failed (Item {item_idx}): Expected row {num_true_clusters-1} (0-indexed) to be non-zero, but it's all zero. " \
                f"num_true_clusters = {num_true_clusters}. Row content: {current_C_padded_np[num_true_clusters-1, :]}"
        if num_true_clusters < max_true_clusters:
            assert np.all(current_C_padded_np[num_true_clusters, :] == 0), \
                f"Assertion Failed (Item {item_idx}): Expected row {num_true_clusters} (0-indexed) to be all zero (padding), but it's not. " \
                f"num_true_clusters = {num_true_clusters}. Row content: {current_C_padded_np[num_true_clusters, :]}"
        if num_true_clusters == 0 and max_true_clusters > 0:
                assert np.all(current_C_padded_np[0, :] == 0), \
                    f"Assertion Failed (Item {item_idx}): num_true_clusters is 0, but the first row is not all zero. " \
                    f"Row content: {current_C_padded_np[0, :]}"

        all_C_padded_list.append(torch.from_numpy(current_C_padded_np))

        num_actual_hits_this_item = len(generated_hits_for_batch_item)
        all_true_num_hits_list.append(min(num_actual_hits_this_item, max_hits_per_batch_item))

        current_H_padded_np = np.zeros((max_hits_per_batch_item, hit_pos_feature_dim), dtype=np.float32)
        actual_hits_to_store_in_tensor = min(num_actual_hits_this_item, max_hits_per_batch_item)

        if actual_hits_to_store_in_tensor > 0:
            hits_array = np.array(generated_hits_for_batch_item)
            current_H_padded_np[:actual_hits_to_store_in_tensor, :] = hits_array[:actual_hits_to_store_in_tensor, :]

        all_H_padded_list.append(torch.from_numpy(current_H_padded_np))

    H_batch = torch.stack(all_H_padded_list)
    C_batch = torch.stack(all_C_padded_list)
    true_k_for_batch_tensor = torch.tensor(all_true_k_list, dtype=torch.long)
    true_num_hits_for_item_tensor = torch.tensor(all_true_num_hits_list, dtype=torch.long)

    return H_batch, C_batch, true_k_for_batch_tensor, true_num_hits_for_item_tensor


# In[3]:

# --- Start of VoxelTransformerClusterer Definition ---
class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class VoxelTransformerClusterer(nn.Module):
    def __init__(self,
                 n_clusters, # K_max: number of output clusters
                 hit_feature_dim, # Dimension of input hit features (used for voxel grid)
                 voxel_dim, # Dimension of features after aggregating hits in a voxel
                 cluster_dim, # Final output dimension for the K_max clusters
                 grid_size,
                 hit_embed_mlp_hidden_dims=None,
                 transformer_model_dim=128,
                 transformer_nhead=4,
                 transformer_nlayers=3,
                 transformer_dim_feedforward=512,
                 transformer_dropout=0.1,
                 max_voxels_per_item=512,
                 random_seed=None,
                 spatial_dims_for_grid=3): # Number of initial features in x_hits to use for spatial voxelization
        super().__init__()
        
        if not TORCH_GEOMETRIC_STACK_AVAILABLE:
            raise ImportError("torch_cluster and/or torch_scatter not found. "
                              "VoxelTransformerClusterer requires these libraries.")

        self.K_max = n_clusters
        self.hit_feature_dim = hit_feature_dim
        self.voxel_dim = voxel_dim
        self.cluster_dim = cluster_dim
        self.grid_size = float(grid_size) # Ensure it's float
        self.transformer_model_dim = transformer_model_dim
        self.max_voxels_per_item_for_padding = max_voxels_per_item
        self.spatial_dims_for_grid = min(spatial_dims_for_grid, hit_feature_dim)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

        if hit_embed_mlp_hidden_dims is None:
            hit_embed_mlp_hidden_dims = [max(voxel_dim, hit_feature_dim)] 
        self.hit_embed_mlp = _MLP(hit_feature_dim, hit_embed_mlp_hidden_dims, voxel_dim)

        self.grid_start_coord = -1.0
        self.grid_end_coord = 1.0
        
        # Calculate num_cells_per_dim only for the spatial dimensions used for the grid
        self.num_cells_per_spatial_dim_val = int(torch.floor(
            torch.tensor((self.grid_end_coord - self.grid_start_coord) / self.grid_size)
        ).item() + 1)

        # Create a tensor representing num_cells for each of the D_grid_features dimensions
        self.num_cells_per_dim_tensor = torch.full(
            (self.spatial_dims_for_grid,), self.num_cells_per_spatial_dim_val, dtype=torch.long
        )
        self.max_local_voxel_id_val = self.num_cells_per_dim_tensor.prod().item()


        self.voxel_project_mlp = nn.Linear(voxel_dim, transformer_model_dim) if voxel_dim != transformer_model_dim else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_model_dim, nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
        self.cluster_prototypes = nn.Parameter(torch.randn(1, self.K_max, transformer_model_dim))
        self.cluster_attention_pooler = nn.MultiheadAttention(
            embed_dim=transformer_model_dim, num_heads=transformer_nhead,
            dropout=transformer_dropout, batch_first=True
        )
        self.final_cluster_projection_mlp = nn.Linear(transformer_model_dim, cluster_dim) if transformer_model_dim != cluster_dim else nn.Identity()

    def _get_voxel_indices_and_scalar_ids(self, hits_spatial_coords):
        # hits_spatial_coords: (N_total_active, D_spatial) in [-1, 1]
        # D_spatial is self.spatial_dims_for_grid
        
        voxel_coords_int = torch.floor(
            (hits_spatial_coords - self.grid_start_coord) / self.grid_size
        ).long()
        
        # Clamp indices to be [0, num_cells_per_dim_val - 1]
        voxel_coords_int = torch.clamp(voxel_coords_int, 0, self.num_cells_per_spatial_dim_val - 1)
        
        # Convert D-dimensional integer voxel coordinates to a scalar ID
        num_cells_per_dim_expanded = self.num_cells_per_dim_tensor.to(hits_spatial_coords.device)
        multipliers = torch.cumprod(
            torch.cat([torch.tensor([1], device=hits_spatial_coords.device), num_cells_per_dim_expanded[:-1]]),
            dim=0
        )
        local_scalar_voxel_ids = (voxel_coords_int * multipliers).sum(dim=1)
        return local_scalar_voxel_ids

    def forward(self, x_hits, num_hits_per_item):
        B, N_max, D_hit_feat_unused = x_hits.shape
        device = x_hits.device
        dtype = x_hits.dtype

        active_hits_list = []
        flat_batch_idx_list = []
        for i in range(B):
            n_actual = num_hits_per_item[i].item()
            if n_actual > 0:
                active_hits_list.append(x_hits[i, :n_actual, :])
                flat_batch_idx_list.append(torch.full((n_actual,), i, device=device, dtype=torch.long))

        if not active_hits_list:
            return torch.zeros(B, self.K_max, self.cluster_dim, device=device, dtype=dtype)

        flat_hits = torch.cat(active_hits_list, dim=0)
        flat_batch_idx = torch.cat(flat_batch_idx_list, dim=0)

        if flat_hits.numel() == 0:
             return torch.zeros(B, self.K_max, self.cluster_dim, device=device, dtype=dtype)

        hits_spatial_coords = flat_hits[:, :self.spatial_dims_for_grid]
        local_voxel_id_per_hit = self._get_voxel_indices_and_scalar_ids(hits_spatial_coords)
        
        global_voxel_scatter_target_ids = flat_batch_idx * self.max_local_voxel_id_val + local_voxel_id_per_hit
        processed_hits = self.hit_embed_mlp(flat_hits)

        unique_global_target_ids, inverse_map = torch.unique(global_voxel_scatter_target_ids, return_inverse=True)
        
        summed_voxel_features = torch_scatter.scatter(
            processed_hits, inverse_map, dim=0, reduce='sum', dim_size=unique_global_target_ids.size(0)
        )
        voxel_hit_counts = torch_scatter.scatter(
            torch.ones_like(inverse_map, dtype=dtype).unsqueeze(-1),
            inverse_map, dim=0, reduce='sum', dim_size=unique_global_target_ids.size(0)
        )
        mean_voxel_features_flat = summed_voxel_features / voxel_hit_counts.clamp(min=1.0)
        batch_idx_for_flat_voxels = unique_global_target_ids // self.max_local_voxel_id_val

        padded_voxel_features = torch.zeros(
            B, self.max_voxels_per_item_for_padding, self.voxel_dim, device=device, dtype=dtype
        )
        src_key_padding_mask = torch.ones(
            B, self.max_voxels_per_item_for_padding, device=device, dtype=torch.bool
        )

        for b_idx in range(B):
            item_mask = (batch_idx_for_flat_voxels == b_idx)
            current_item_voxel_features = mean_voxel_features_flat[item_mask]
            num_voxels_for_this_item = current_item_voxel_features.shape[0]

            if num_voxels_for_this_item > 0:
                len_to_copy = min(num_voxels_for_this_item, self.max_voxels_per_item_for_padding)
                padded_voxel_features[b_idx, :len_to_copy] = current_item_voxel_features[:len_to_copy]
                src_key_padding_mask[b_idx, :len_to_copy] = False
        
        tf_input_voxels = self.voxel_project_mlp(padded_voxel_features)
        encoded_voxels = self.transformer_encoder(
            tf_input_voxels, src_key_padding_mask=src_key_padding_mask
        )
        cluster_protos_b = self.cluster_prototypes.repeat(B, 1, 1)
        attended_clusters, _ = self.cluster_attention_pooler(
            query=cluster_protos_b, key=encoded_voxels, value=encoded_voxels,
            key_padding_mask=src_key_padding_mask
        )
        output_centroids = self.final_cluster_projection_mlp(attended_clusters)
        return output_centroids


class AttentionLayer(nn.Module):
    def __init__(self, k_max, query_dim, hit_feature_dim, head_dim, value_dim, num_heads=1, mlp_hidden_dim_multiplier=2):
        super().__init__()
        self.K_max = k_max
        self.query_dim = query_dim
        self.hit_feature_dim = hit_feature_dim
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.num_heads = num_heads

        self.norm_slots = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, head_dim * num_heads, bias=False)
        self.to_k = nn.Linear(hit_feature_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hit_feature_dim, value_dim * num_heads, bias=False)
        self.scale = head_dim ** -0.5

        self.mlp = nn.Sequential(
            nn.Linear(value_dim * num_heads, query_dim * mlp_hidden_dim_multiplier),
            nn.ReLU(),
            nn.Linear(query_dim * mlp_hidden_dim_multiplier, query_dim)
        )
        self.norm_mlp_output = nn.LayerNorm(query_dim)

    def forward(self, slot_representations, hit_features):
        B, N, _ = hit_features.shape
        _, K, D_query = slot_representations.shape
        slots_norm = self.norm_slots(slot_representations)
        q = self.to_q(slots_norm)
        k = self.to_k(hit_features)
        v = self.to_v(hit_features)
        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.value_dim).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(B, K, self.num_heads * self.value_dim)
        mlp_out = self.mlp(attended_values)
        updated_slots = self.norm_mlp_output(slot_representations + mlp_out)
        return updated_slots

class CrossAttentionClusteringLearnableK(nn.Module):
    def __init__(self, k_max, hit_feature_dim, query_dim, head_dim, value_dim, cluster_feature_dim,
                 num_attention_layers=1,
                 num_heads=1, mlp_hidden_dim_multiplier=2,
                 existence_logit_bias_init=-2.0,
                 use_dbscan: bool = False,
                 dbscan_eps: float = 0.05,
                 dbscan_min_samples: int = 5,
                 # --- Start of VoxelTransformer HParams for __init__ ---
                 use_voxel_transformer: bool = False,
                 vt_grid_size: float = 0.1,
                 vt_hit_embed_mlp_hidden_dims=None, # e.g. [64]
                 vt_voxel_dim=None, # if None, will be query_dim
                 vt_transformer_model_dim=128,
                 vt_transformer_nhead=4,
                 vt_transformer_nlayers=2,
                 vt_transformer_dim_feedforward=256,
                 vt_transformer_dropout=0.1,
                 vt_max_voxels_per_item=256,
                 vt_spatial_dims_for_grid=3
                 # --- End of VoxelTransformer HParams ---
                ):
        super().__init__()
        self.K_max = k_max
        self.query_dim = query_dim # This is the dimension expected by AttentionLayers for slots
        self.cluster_feature_dim = cluster_feature_dim # Final output dim of the model
        self.hit_feature_dim = hit_feature_dim

        self.use_dbscan = use_dbscan
        self.use_voxel_transformer = use_voxel_transformer

        if self.use_dbscan and self.use_voxel_transformer:
            raise ValueError("Only one of use_dbscan or use_voxel_transformer can be True.")

        # DBSCAN specific setup
        if self.use_dbscan:
            if not CUML_AVAILABLE or cuMLDBSCAN is None:
                 raise RuntimeError("use_dbscan is True, but cuML is not available.")
            self.dbscan_eps_val = dbscan_eps
            self.dbscan_min_samples_val = dbscan_min_samples
            if self.hit_feature_dim != self.query_dim:
                self.dbscan_feature_projection = nn.Linear(self.hit_feature_dim, self.query_dim)
            else:
                self.dbscan_feature_projection = nn.Identity()
            self.dbscan_model = cuMLDBSCAN(
                eps=self.dbscan_eps_val,
                min_samples=self.dbscan_min_samples_val,
                output_type='cudf'
            )
        
        # VoxelTransformerClusterer specific setup
        if self.use_voxel_transformer:
            if not TORCH_GEOMETRIC_STACK_AVAILABLE:
                 raise RuntimeError("use_voxel_transformer is True, but torch_cluster/torch_scatter are not available.")
            
            # The VoxelTransformerClusterer should output features of query_dim to be used as slots
            # So, its internal cluster_dim argument will be self.query_dim
            _voxel_dim = vt_voxel_dim if vt_voxel_dim is not None else self.query_dim

            self.voxel_transformer_clusterer = VoxelTransformerClusterer(
                n_clusters=self.K_max, # It should produce K_max slot initializations
                hit_feature_dim=self.hit_feature_dim,
                voxel_dim=_voxel_dim, # Intermediate dim for aggregated voxel features
                cluster_dim=self.query_dim, # Final output of VTC should match query_dim for slots
                grid_size=vt_grid_size,
                hit_embed_mlp_hidden_dims=vt_hit_embed_mlp_hidden_dims,
                transformer_model_dim=vt_transformer_model_dim,
                transformer_nhead=vt_transformer_nhead,
                transformer_nlayers=vt_transformer_nlayers,
                transformer_dim_feedforward=vt_transformer_dim_feedforward,
                transformer_dropout=vt_transformer_dropout,
                max_voxels_per_item=vt_max_voxels_per_item,
                spatial_dims_for_grid=vt_spatial_dims_for_grid
            )
            
        # Default learnable queries (used if neither DBSCAN nor VoxelTransformer is active)
        self.cluster_queries_init = nn.Parameter(torch.randn(1, self.K_max, query_dim))
        self.query_pos_embed = nn.Parameter(torch.randn(1, self.K_max, query_dim))

        self.attention_layers = nn.ModuleList([
            AttentionLayer(k_max, query_dim, hit_feature_dim, head_dim, value_dim,
                           num_heads, mlp_hidden_dim_multiplier)
            for _ in range(num_attention_layers)
        ])

        ffn_output_dim = self.cluster_feature_dim + 1 
        self.final_ffn = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim * mlp_hidden_dim_multiplier),
            nn.ReLU(),
            nn.Linear(query_dim * mlp_hidden_dim_multiplier, ffn_output_dim)
        )

        if existence_logit_bias_init is not None:
            with torch.no_grad():
                self.final_ffn[-1].bias[self.cluster_feature_dim].fill_(existence_logit_bias_init)

    def forward(self, x_hits, num_hits_per_item=None):
        B, N_max_hits_in_batch, D_feat_x_hits = x_hits.shape
        
        if self.use_voxel_transformer:
            if num_hits_per_item is None:
                raise ValueError("num_hits_per_item must be provided for VoxelTransformerClusterer.")
            # VoxelTransformerClusterer output is (B, K_max, query_dim)
            slot_repr = self.voxel_transformer_clusterer(x_hits, num_hits_per_item)

        elif self.use_dbscan:
            if num_hits_per_item is None:
                raise ValueError("num_hits_per_item must be provided when use_dbscan is True.")
            
            # DBSCAN implementation
            # (Output: projected centroids of shape (B, K_max, query_dim))
            temp_slot_repr = torch.zeros(B, self.K_max, self.query_dim, device=x_hits.device, dtype=x_hits.dtype)
            for i in range(B):
                n_hits_item_i = num_hits_per_item[i].item()
                if n_hits_item_i == 0: continue
                current_hits_for_dbscan = x_hits[i, :n_hits_item_i, :]
                if current_hits_for_dbscan.shape[0] < self.dbscan_min_samples_val:
                    if current_hits_for_dbscan.shape[0] > 0 and self.K_max > 0:
                        centroid_tensor = current_hits_for_dbscan.mean(axis=0)
                        projected_centroid = self.dbscan_feature_projection(centroid_tensor)
                        temp_slot_repr[i, 0, :] = projected_centroid
                    continue
                try:
                    labels_gdf_series = self.dbscan_model.fit_predict(current_hits_for_dbscan)
                    labels_torch_gpu = torch.as_tensor(labels_gdf_series.values, device=x_hits.device)
                except Exception as e:
                    if current_hits_for_dbscan.shape[0] > 0 and self.K_max > 0:
                        centroid_tensor = current_hits_for_dbscan.mean(axis=0)
                        projected_centroid = self.dbscan_feature_projection(centroid_tensor)
                        temp_slot_repr[i, 0, :] = projected_centroid
                    continue

                unique_labels_gpu = torch.unique(labels_torch_gpu)
                dbscan_cluster_centroids_list = []
                for k_label_tensor in unique_labels_gpu:
                    k_label_val = k_label_tensor.item()
                    if k_label_val == -1: continue
                    mask_torch_gpu = (labels_torch_gpu == k_label_tensor)
                    if torch.any(mask_torch_gpu):
                        cluster_hits_gpu = current_hits_for_dbscan[mask_torch_gpu]
                        centroid_tensor_gpu = cluster_hits_gpu.mean(axis=0)
                        dbscan_cluster_centroids_list.append(centroid_tensor_gpu)
                
                num_found_dbscan_clusters = len(dbscan_cluster_centroids_list)
                num_slots_to_fill_with_dbscan = min(num_found_dbscan_clusters, self.K_max)
                for j in range(num_slots_to_fill_with_dbscan):
                    centroid_feature_tensor = dbscan_cluster_centroids_list[j] # (hit_feature_dim)
                    temp_slot_repr[i, j, :] = self.dbscan_feature_projection(centroid_feature_tensor)
            slot_repr = temp_slot_repr
        
        else: # Original learnable slot initialization
            slot_repr = self.cluster_queries_init + self.query_pos_embed
            slot_repr = slot_repr.repeat(B, 1, 1)
        
        for layer_idx, layer in enumerate(self.attention_layers):
            slot_repr = layer(slot_repr, x_hits)

        raw_output = self.final_ffn(slot_repr)
        predicted_cluster_features = raw_output[..., :self.cluster_feature_dim]
        existence_logits = raw_output[..., self.cluster_feature_dim:]

        return predicted_cluster_features, existence_logits


def compute_diversity_loss_kernel_energy(pred_features, pred_logits, geomloss_sigma_diversity=0.1, epsilon=1e-8):
    B, K_max, F_out = pred_features.shape
    device = pred_features.device

    if K_max <= 1:
        return torch.tensor(0.0, device=device)

    existence_probs = torch.sigmoid(pred_logits).squeeze(-1)
    prob_sum_per_item = existence_probs.sum(dim=1, keepdim=True).clamp(min=epsilon) # Clamp sum to avoid div by zero
    weights = existence_probs / prob_sum_per_item 
    K_xx = geomloss_gaussian_kernel(pred_features, pred_features, blur=geomloss_sigma_diversity)
    energy_batched = torch.einsum('bi,bij,bj->b', weights, K_xx, weights)
    return energy_batched.mean()

def compute_cluster_loss_geomloss(
    pred_features,  # (B, K_max_pred, F_out)
    pred_logits,    # (B, K_max_pred, 1)
    gt_cluster_matrix, # (B, K_max_gt, F_out)
    num_gt_clusters_per_item, # (B,)
    existence_loss_fn_sum_reduction,
    lambda_diversity=0.0,
    geomloss_sigma_diversity=0.1,
    epsilon_diversity_weight_norm=1e-8, 
    cost_type='L1',        
    geomloss_blur=0.1,      
    geomloss_nits=10,       
):
    batch_size, k_max_pred, feature_dim = pred_features.shape
    _, k_max_gt, _ = gt_cluster_matrix.shape
    device = pred_features.device

    alpha_weights = torch.sigmoid(pred_logits.squeeze(-1)) 
    gt_indices = torch.arange(k_max_gt, device=device).unsqueeze(0) 
    num_gt_expanded = num_gt_clusters_per_item.unsqueeze(1) 
    beta_weights = (gt_indices < num_gt_expanded).float() 

    if cost_type == 'L1':
        p_norm = 1
    elif cost_type == 'L2_sq':
        p_norm = 2 
    else:
        raise ValueError(f"Unsupported cost_type for geomloss: {cost_type}. Use 'L1' or 'L2_sq'.")

    geomloss_feature_fn = SamplesLoss(
        loss="sinkhorn", 
        p=p_norm, 
        blur=geomloss_blur, 
        debias=False, 
        scaling=0.75, 
    )
    
    feature_loss_values_per_item = geomloss_feature_fn(
        alpha_weights, pred_features, 
        beta_weights, gt_cluster_matrix
    ) 

    total_feature_loss = feature_loss_values_per_item.sum()
    num_total_actual_gt_clusters = num_gt_clusters_per_item.sum().float().clamp(min=1.0)
    avg_feature_loss = total_feature_loss / num_total_actual_gt_clusters

    target_existence_hungarian = torch.zeros_like(pred_logits, device=device) 
    with torch.no_grad():
        if cost_type == 'L1':
            cost_matrix_C_hung = torch.cdist(pred_features.detach(), gt_cluster_matrix, p=1)
        elif cost_type == 'L2_sq': 
            cost_matrix_C_hung = torch.cdist(pred_features.detach(), gt_cluster_matrix, p=2).pow(2)
        else: 
            cost_matrix_C_hung = torch.cdist(pred_features.detach(), gt_cluster_matrix, p=2).pow(2)


    for i in range(batch_size):
        k_gt_i = num_gt_clusters_per_item[i].item()
        if k_gt_i > 0:
            cost_m_item_tensor = cost_matrix_C_hung[i, :, :k_gt_i]
            cost_m_item_np = cost_m_item_tensor.cpu().numpy()
            try:
                row_ind, col_ind = linear_sum_assignment(cost_m_item_np)
                if row_ind.size > 0:
                    row_ind_tensor = torch.from_numpy(row_ind).long().to(device)
                    target_existence_hungarian[i, row_ind_tensor, 0] = 1.0
            except ValueError as e:
                continue
    
    total_existence_loss = existence_loss_fn_sum_reduction(pred_logits, target_existence_hungarian)
    avg_existence_loss = total_existence_loss / (batch_size * k_max_pred) if (batch_size * k_max_pred) > 0 else torch.tensor(0.0, device=device)

    div_loss_value = torch.tensor(0.0, device=device)
    if lambda_diversity > 0 and k_max_pred > 1:
        div_loss_value = compute_diversity_loss_kernel_energy(
            pred_features, pred_logits, 
            geomloss_sigma_diversity=geomloss_sigma_diversity,
            epsilon=epsilon_diversity_weight_norm 
        )
    
    return avg_feature_loss, avg_existence_loss, div_loss_value


# --- Plotting functions for TensorBoard (Hungarian-based for eval) ---
# ... (Plotting functions remain unchanged, omitted for brevity) ...
def plot_matched_scatter_to_tensorboard(writer, tag_prefix, step,
                                        hits_np, true_clusters_np, pred_features_np, pred_probs_np,
                                        row_ind, col_ind, 
                                        item_idx, F_cluster_out_gt, existence_threshold):
    fig, ax = plt.subplots(figsize=(8, 8))
    if hits_np.shape[0] > 0 and hits_np.shape[1] >= 2:
        ax.scatter(hits_np[:, 0], hits_np[:, 1], marker=".", label="Hits", alpha=0.3, s=10, zorder=1, color='gray')
    if true_clusters_np.shape[0] > 0 and true_clusters_np.shape[1] >= 2:
        ax.scatter(true_clusters_np[:, 0], true_clusters_np[:, 1], marker="o", s=150, facecolors='none', edgecolors='blue', linewidth=2, label="True Clusters", zorder=2)
    active_pred_mask = (pred_probs_np > existence_threshold).squeeze(-1) 
    active_pred_features = pred_features_np[active_pred_mask]
    active_pred_probs = pred_probs_np[active_pred_mask]
    if active_pred_features.shape[0] > 0 and active_pred_features.shape[1] >= 2:
        scatter_plot_data = active_pred_probs.squeeze() if active_pred_probs.ndim > 1 else active_pred_probs
        scatter = ax.scatter(active_pred_features[:, 0], active_pred_features[:, 1], marker="x", s=150,
                             c=scatter_plot_data, cmap='Reds', vmin=0.0, vmax=1.0,
                             linewidth=2, label="Predicted Clusters (Active)", zorder=3)
        fig.colorbar(scatter, ax=ax, label="Existence Probability")
    if row_ind.size > 0 and true_clusters_np.shape[0] > 0 and pred_features_np.shape[0] > 0:
        for r_idx, c_idx in zip(row_ind, col_ind):
            if r_idx < pred_features_np.shape[0] and c_idx < true_clusters_np.shape[0]: 
                pred_pt = pred_features_np[r_idx, :2] 
                gt_pt = true_clusters_np[c_idx, :2]   
                is_active = pred_probs_np[r_idx].item() > existence_threshold if r_idx < pred_probs_np.shape[0] else False
                line_color = 'green' if is_active else 'lightgrey'
                line_style = '-' if is_active else '--'
                ax.plot([pred_pt[0], gt_pt[0]], [pred_pt[1], gt_pt[1]], color=line_color, linestyle=line_style, alpha=0.7, linewidth=1.5, zorder=2.5)
    ax.set_title(f"Item {item_idx} - Matched Clusters (Step {step})")
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlabel("Feature Dim 1")
    ax.set_ylabel("Feature Dim 2")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    all_points_x = []
    all_points_y = []
    if hits_np.shape[0] > 0 and hits_np.shape[1] >=2:
        all_points_x.extend(hits_np[:, 0])
        all_points_y.extend(hits_np[:, 1])
    if true_clusters_np.shape[0] > 0 and true_clusters_np.shape[1] >=2:
        all_points_x.extend(true_clusters_np[:, 0])
        all_points_y.extend(true_clusters_np[:, 1])
    if active_pred_features.shape[0] > 0 and active_pred_features.shape[1] >=2:
        all_points_x.extend(active_pred_features[:, 0])
        all_points_y.extend(active_pred_features[:, 1])
    if all_points_x and all_points_y:
        min_x, max_x = np.min(all_points_x), np.max(all_points_x)
        min_y, max_y = np.min(all_points_y), np.max(all_points_y)
        padding_x = 0.1 * abs(max_x-min_x) + 0.1
        padding_y = 0.1 * abs(max_y-min_y) + 0.1
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
    else: 
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    writer.add_figure(f"{tag_prefix}/Matched_Scatter_Item_{item_idx}", fig, global_step=step)
    plt.close(fig)

def plot_tsne_features_to_tensorboard(writer, tag_prefix, step,
                                      true_features_np, pred_features_all_np, pred_probs_all_np,
                                      item_idx, existence_threshold, F_cluster_out_gt, n_components=2, perplexity=10):
    if F_cluster_out_gt < n_components: 
        return
    active_pred_mask = (pred_probs_all_np > existence_threshold).squeeze(-1)
    pred_features_active_np = pred_features_all_np[active_pred_mask]
    if true_features_np.shape[0] == 0 and pred_features_active_np.shape[0] == 0:
        return
    n_true = true_features_np.shape[0]
    n_pred_active = pred_features_active_np.shape[0]
    combined_features = []
    labels = [] 
    if n_true > 0:
        combined_features.append(true_features_np)
        labels.extend([0] * n_true)
    if n_pred_active > 0:
        combined_features.append(pred_features_active_np)
        labels.extend([1] * n_pred_active)
    if not combined_features: 
        return
    combined_features_np = np.concatenate(combined_features, axis=0)
    current_perplexity = min(perplexity, combined_features_np.shape[0] - 1)
    if combined_features_np.shape[0] <= current_perplexity or current_perplexity <=0 :
        if F_cluster_out_gt >= 2 and combined_features_np.shape[0] > 0: # Check if features are at least 2D for scatter
            fig, ax = plt.subplots(figsize=(8, 8))
            if n_true > 0 and true_features_np.shape[1] >=2 :
                ax.scatter(true_features_np[:, 0], true_features_np[:, 1], c='blue', label=f'True Clusters ({n_true})', marker='o', s=60, alpha=0.7)
            if n_pred_active > 0 and pred_features_active_np.shape[1] >=2:
                ax.scatter(pred_features_active_np[:, 0], pred_features_active_np[:, 1], c='red', label=f'Predicted Active ({n_pred_active})', marker='x', s=60, alpha=0.7)
            ax.set_title(f"Item {item_idx} - Feature Space (Raw 2D, Step {step})")
            ax.legend()
            writer.add_figure(f"{tag_prefix}/Feature_Space_Raw_Item_{item_idx}", fig, global_step=step)
            plt.close(fig)
        return
    tsne = TSNE(n_components=n_components, random_state=0, perplexity=current_perplexity, init='pca', learning_rate='auto')
    embedded_features = tsne.fit_transform(combined_features_np)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['blue', 'red']
    markers = ['o', 'x']
    sizes = [60, 60]
    current_idx = 0
    if n_true > 0:
        ax.scatter(embedded_features[current_idx:current_idx+n_true, 0],
                   embedded_features[current_idx:current_idx+n_true, 1],
                   c=colors[0], label=f'True Clusters ({n_true})', marker=markers[0], s=sizes[0], alpha=0.7)
        current_idx += n_true
    if n_pred_active > 0:
        ax.scatter(embedded_features[current_idx:current_idx+n_pred_active, 0],
                   embedded_features[current_idx:current_idx+n_pred_active, 1],
                   c=colors[1], label=f'Predicted Active ({n_pred_active})', marker=markers[1], s=sizes[1], alpha=0.7)
    ax.set_title(f"Item {item_idx} - t-SNE of Features (Step {step})")
    ax.legend()
    writer.add_figure(f"{tag_prefix}/tSNE_Features_Item_{item_idx}", fig, global_step=step)
    plt.close(fig)

def plot_slot_usage_to_tensorboard(writer, tag_prefix, step, slot_match_counts_np, k_max):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(k_max), slot_match_counts_np, color='teal')
    ax.set_xlabel("Slot Index")
    ax.set_ylabel("Number of Times Matched to a True Cluster")
    ax.set_title(f"Slot Matching Frequency (Step {step})")
    ax.set_xticks(np.arange(0, k_max, max(1, k_max // 16))) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    writer.add_figure(f"{tag_prefix}/Slot_Usage_Frequency", fig, global_step=step)
    plt.close(fig)

def plot_existence_prob_histograms_to_tensorboard(writer, tag_prefix, step,
                                                 probs_of_matched_slots_np,
                                                 probs_of_unmatched_active_slots_np):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    if probs_of_matched_slots_np.size > 0:
        ax[0].hist(probs_of_matched_slots_np, bins=20, range=(0,1), color='green', alpha=0.7, label='Matched Slots')
    ax[0].set_title("Probs of Matched Slots")
    ax[0].set_xlabel("Existence Probability")
    ax[0].set_ylabel("Frequency")
    ax[0].grid(True, linestyle='--', alpha=0.6)
    if probs_of_unmatched_active_slots_np.size > 0:
        ax[1].hist(probs_of_unmatched_active_slots_np, bins=20, range=(0,1), color='orange', alpha=0.7, label='Unmatched Active Slots')
    ax[1].set_title("Probs of Unmatched Active Slots")
    ax[1].set_xlabel("Existence Probability")
    ax[1].grid(True, linestyle='--', alpha=0.6)
    fig.suptitle(f"Existence Probability Distributions (Step {step})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    writer.add_figure(f"{tag_prefix}/Existence_Prob_Histograms", fig, global_step=step)
    plt.close(fig)

def plot_k_confusion_matrix_to_tensorboard(writer, tag_prefix, step, k_pairs, k_max_possible_clusters):
    conf_matrix_size = k_max_possible_clusters + 1 
    conf_matrix = np.zeros((conf_matrix_size, conf_matrix_size), dtype=int)
    for true_k, pred_k in k_pairs:
        true_k_clamped = min(true_k, k_max_possible_clusters)
        pred_k_clamped = min(pred_k, k_max_possible_clusters)
        conf_matrix[true_k_clamped, pred_k_clamped] += 1
    fig, ax = plt.subplots(figsize=(max(8, conf_matrix_size*0.5), max(6, conf_matrix_size*0.4)))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, label='Number of Items')
    tick_labels = [str(i) for i in range(conf_matrix_size)]
    ax.set_xlabel("Predicted Number of Clusters (Pred K)", fontsize=10)
    ax.set_ylabel("True Number of Clusters (True K)", fontsize=10)
    ax.set_title(f"Confusion Matrix for Cluster Counts (Step {step})", fontsize=12)
    tick_step = 1
    if conf_matrix_size > 10: tick_step = 2
    if conf_matrix_size > 20: tick_step = 5
    selected_ticks = np.arange(0, conf_matrix_size, tick_step)
    selected_tick_labels = [str(i) for i in selected_ticks]
    ax.set_xticks(selected_ticks)
    ax.set_xticklabels(selected_tick_labels)
    ax.set_yticks(selected_ticks)
    ax.set_yticklabels(selected_tick_labels)
    ax.tick_params(axis='both', which='major', labelsize=8)
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if conf_matrix[i, j] > 0: 
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black",
                        fontsize=8)
    fig.tight_layout()
    writer.add_figure(f"{tag_prefix}/K_Number_Confusion_Matrix", fig, global_step=step)
    plt.close(fig)

def log_gradients_to_tensorboard(writer, model, epoch_step, tag_prefix="Gradients"):
    # ... (Gradient logging function remains unchanged, omitted for brevity) ...
    all_finite_grads = []
    total_nan_grads = 0
    total_inf_grads = 0
    overall_max_abs_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_data = param.grad.detach()
            nan_count = torch.isnan(grad_data).sum().item()
            inf_count = torch.isinf(grad_data).sum().item()
            total_nan_grads += nan_count
            total_inf_grads += inf_count
            finite_grads = grad_data[torch.isfinite(grad_data)]
            if finite_grads.numel() > 0:
                all_finite_grads.append(finite_grads.flatten())
                current_max_abs = finite_grads.abs().max().item()
                if current_max_abs > overall_max_abs_grad:
                    overall_max_abs_grad = current_max_abs
                writer.add_histogram(f"{tag_prefix}_PerLayer/{name}/Gradient_Distribution", finite_grads, epoch_step)
                writer.add_scalar(f"{tag_prefix}_PerLayer/{name}/Max_Abs_Gradient", current_max_abs, epoch_step)
                writer.add_scalar(f"{tag_prefix}_PerLayer/{name}/L2_Norm", torch.norm(finite_grads, p=2).item(), epoch_step)
            writer.add_scalar(f"{tag_prefix}_PerLayer/{name}/NaN_Count", nan_count, epoch_step)
            writer.add_scalar(f"{tag_prefix}_PerLayer/{name}/Inf_Count", inf_count, epoch_step)
            writer.add_scalar(f"{tag_prefix}_PerLayer/{name}/Total_Elements", grad_data.numel(), epoch_step)
    writer.add_scalar(f"{tag_prefix}/Total_NaN_Gradients", total_nan_grads, epoch_step)
    writer.add_scalar(f"{tag_prefix}/Total_Inf_Gradients", total_inf_grads, epoch_step)
    writer.add_scalar(f"{tag_prefix}/Overall_Max_Abs_Gradient", overall_max_abs_grad, epoch_step)
    if all_finite_grads:
        concatenated_finite_grads = torch.cat(all_finite_grads)
        if concatenated_finite_grads.numel() > 0: 
            writer.add_histogram(f"{tag_prefix}/All_Gradients_Distribution", concatenated_finite_grads, epoch_step)
            writer.add_scalar(f"{tag_prefix}/Overall_L2_Norm_Gradients", torch.norm(concatenated_finite_grads, p=2).item(), epoch_step)
        else:
            writer.add_scalar(f"{tag_prefix}/Overall_L2_Norm_Gradients", 0.0, epoch_step) 
    else: 
        writer.add_scalar(f"{tag_prefix}/Overall_L2_Norm_Gradients", 0.0, epoch_step)

# In[4]:

def setup_experiment_configuration(config_path: str):
    """Sets up all hyperparameters, logging, and device."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_params = config['data_params']
    model_config = config['model_params']
    training_hyperparams = config['training_params']
    loss_params_dict = config['loss_params']
    eval_config = config['eval_params']
    profiler_config = config['profiler_params']

    F_in_hits = data_params['F_in_hits']
    F_cluster_out_gt = data_params['F_cluster_out_gt']

    # Construct model_params_dict for initialize_model_and_optimizer
    k_max_factor = model_config.get("k_max_factor_of_max_true_clusters", 1.0)
    k_max_val = int(data_params["max_true_clusters"] * k_max_factor)

    model_params_dict = {
        "k_max": k_max_val,
        "hit_feature_dim": F_in_hits,
        "query_dim": model_config["query_dim_for_slots"],
        "head_dim_total": model_config["head_dim_total"],
        "value_dim_total": model_config["value_dim_total"],
        "cluster_feature_dim": F_cluster_out_gt,
        "num_attention_layers": model_config["num_attention_layers"],
        "num_heads": model_config["num_heads"],
        "mlp_hidden_dim_multiplier": model_config["mlp_hidden_dim_multiplier"],
        "existence_logit_bias_init": model_config.get("existence_logit_bias_init"),
    }

    slot_init_config = model_config.get("slot_initialization", {})
    use_dbscan = slot_init_config.get("use_dbscan", False)
    use_voxel_transformer = slot_init_config.get("use_voxel_transformer", False)

    model_params_dict["use_dbscan"] = use_dbscan
    if use_dbscan:
        dbscan_c = model_config.get("dbscan_params", {})
        model_params_dict["dbscan_eps"] = dbscan_c.get("eps", 0.05)
        model_params_dict["dbscan_min_samples"] = dbscan_c.get("min_samples", 5)

    model_params_dict["use_voxel_transformer"] = use_voxel_transformer
    if use_voxel_transformer:
        vt_c = model_config.get("voxel_transformer_params", {})
        model_params_dict["vt_grid_size"] = vt_c.get("grid_size", 0.1)
        model_params_dict["vt_hit_embed_mlp_hidden_dims"] = vt_c.get("hit_embed_mlp_hidden_dims", [max(vt_c.get("voxel_dim", model_config["query_dim_for_slots"]), F_in_hits)])
        model_params_dict["vt_voxel_dim"] = vt_c.get("voxel_dim") # Can be None, model will default to query_dim
        model_params_dict["vt_transformer_model_dim"] = vt_c.get("transformer_model_dim", 128)
        model_params_dict["vt_transformer_nhead"] = vt_c.get("transformer_nhead", 4)
        model_params_dict["vt_transformer_nlayers"] = vt_c.get("transformer_nlayers", 3)
        model_params_dict["vt_transformer_dim_feedforward"] = vt_c.get("transformer_dim_feedforward", 512)
        model_params_dict["vt_transformer_dropout"] = vt_c.get("transformer_dropout", 0.1)
        model_params_dict["vt_max_voxels_per_item"] = vt_c.get("max_voxels_per_item", 256)
        model_params_dict["vt_spatial_dims_for_grid"] = vt_c.get("spatial_dims_for_grid", 3)
        if model_params_dict["vt_spatial_dims_for_grid"] > F_in_hits:
            raise ValueError(f"vt_spatial_dims_for_grid ({model_params_dict['vt_spatial_dims_for_grid']}) "
                             f"cannot exceed F_in_hits ({F_in_hits}).")

    assert model_params_dict["head_dim_total"] % model_params_dict["num_heads"] == 0, \
        "head_dim_total must be divisible by num_heads"
    assert model_params_dict["value_dim_total"] % model_params_dict["num_heads"] == 0, \
        "value_dim_total must be divisible by num_heads"

    if use_dbscan and use_voxel_transformer:
        raise ValueError("Cannot use both DBSCAN and VoxelTransformer. Choose one.")

    if use_voxel_transformer and not TORCH_GEOMETRIC_STACK_AVAILABLE:
        print("WARNING: 'use_voxel_transformer' is True but torch_cluster/torch_scatter not found. Reverting to False.")
        model_params_dict["use_voxel_transformer"] = False
        use_voxel_transformer = False # Update local flag

    if use_dbscan and (not CUML_AVAILABLE or cuMLDBSCAN is None):
        print("WARNING: 'use_dbscan' is True but cuML not found. Reverting to False.")
        model_params_dict["use_dbscan"] = False
        use_dbscan = False # Update local flag

    eval_specific_params = {
        "existence_threshold_eval": eval_config["existence_threshold"],
        "cost_type_feature_loss_for_hungarian": loss_params_dict["cost_type_feature_loss"]
    }

    mode_tag = ""
    if use_dbscan: mode_tag = "_DBSCAN"
    elif use_voxel_transformer: mode_tag = "_VoxelTF"
    else: mode_tag = "_LearnableQueries"

    log_dir = f"runs/clustering_experiment{mode_tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if (use_dbscan or use_voxel_transformer) and device.type == 'cpu':
        print("WARNING: DBSCAN or VoxelTransformer selected, but device is CPU. GPU is recommended/required. Forcing to CPU-compatible default.")
        model_params_dict["use_dbscan"] = False
        model_params_dict["use_voxel_transformer"] = False
        if use_dbscan: print("DBSCAN usage disabled.")
        if use_voxel_transformer: print("VoxelTransformer usage disabled.")

    return (data_params, F_in_hits, F_cluster_out_gt, model_params_dict,
            training_hyperparams, loss_params_dict, eval_specific_params,
            profiler_config, log_dir, writer, device)

def initialize_model_and_optimizer(model_params_dict, device):
    D_head_per_attn_head = model_params_dict["head_dim_total"] // model_params_dict["num_heads"]
    D_value_per_attn_head = model_params_dict["value_dim_total"] // model_params_dict["num_heads"]

    model = CrossAttentionClusteringLearnableK(
        k_max=model_params_dict["k_max"],
        hit_feature_dim=model_params_dict["hit_feature_dim"],
        query_dim=model_params_dict["query_dim"], # This is the slot dimension
        head_dim=D_head_per_attn_head,
        value_dim=D_value_per_attn_head,
        cluster_feature_dim=model_params_dict["cluster_feature_dim"], # Final output for loss
        num_attention_layers=model_params_dict["num_attention_layers"],
        num_heads=model_params_dict["num_heads"],
        mlp_hidden_dim_multiplier=model_params_dict["mlp_hidden_dim_multiplier"],
        existence_logit_bias_init=model_params_dict["existence_logit_bias_init"],
        
        use_dbscan=model_params_dict.get("use_dbscan", False),
        dbscan_eps=model_params_dict.get("dbscan_eps", 0.05),
        dbscan_min_samples=model_params_dict.get("dbscan_min_samples", 5),
        use_voxel_transformer=model_params_dict.get("use_voxel_transformer", False),
        vt_grid_size=model_params_dict.get("vt_grid_size", 0.1),
        vt_hit_embed_mlp_hidden_dims=model_params_dict.get("vt_hit_embed_mlp_hidden_dims", [64]),
        # Pass vt_voxel_dim directly; if None, CrossAttentionClusteringLearnableK handles default to query_dim
        vt_voxel_dim=model_params_dict.get("vt_voxel_dim"),
        vt_transformer_model_dim=model_params_dict.get("vt_transformer_model_dim", 128),
        vt_transformer_nhead=model_params_dict.get("vt_transformer_nhead", 4),
        vt_transformer_nlayers=model_params_dict.get("vt_transformer_nlayers", 2),
        vt_transformer_dim_feedforward=model_params_dict.get("vt_transformer_dim_feedforward", 256),
        vt_transformer_dropout=model_params_dict.get("vt_transformer_dropout", 0.1),
        vt_max_voxels_per_item=model_params_dict.get("vt_max_voxels_per_item", 256),
        vt_spatial_dims_for_grid=model_params_dict.get("vt_spatial_dims_for_grid", 3)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params_dict["learning_rate"]) # learning_rate is now in model_params_dict
    return model, optimizer

# ... (rest of the script: pregenerate_data_sets, perform_training_step, evaluation functions, etc.
#      remain largely unchanged as they operate on the output of the model's forward pass)
#      Make sure plot_final_loss_summary, run_training_pipeline, and the main execution block
#      are present as in slot_attention(1).py
def pregenerate_data_sets(data_params, F_cluster_out_gt, F_in_hits, num_pregen_train_batches, device):
    # ... (Unchanged from slot_attention(1).py)
    print("Testing data generation once with assertions...")
    _, _, _, _ = generate_pseudo_data(
        max_true_clusters=data_params["max_true_clusters"],
        max_hits_per_batch_item=data_params["max_hits_per_batch_item"],
        avg_true_clusters=data_params["avg_true_clusters"],
        avg_hits_per_cluster=data_params["avg_hits_per_cluster"],
        batch_size=data_params["batch_size"],
        cluster_pos_feature_dim=F_cluster_out_gt,
        hit_pos_feature_dim=F_in_hits,
        ensure_min_one_hit_per_cluster=data_params["ensure_min_one_hit_per_cluster"]
    )
    print("Data generation test passed assertions.")

    print(f"\nPre-generating {num_pregen_train_batches} training batches...")
    pregen_train_data = []
    for _ in range(num_pregen_train_batches):
        h, tc, tk, tnh = generate_pseudo_data(
            max_true_clusters=data_params["max_true_clusters"],
            max_hits_per_batch_item=data_params["max_hits_per_batch_item"],
            avg_true_clusters=data_params["avg_true_clusters"],
            avg_hits_per_cluster=data_params["avg_hits_per_cluster"],
            batch_size=data_params["batch_size"],
            cluster_pos_feature_dim=F_cluster_out_gt,
            hit_pos_feature_dim=F_in_hits,
            ensure_min_one_hit_per_cluster=data_params["ensure_min_one_hit_per_cluster"]
        )
        pregen_train_data.append({
            "x_hits_input": h.to(device),
            "gt_c_matrix": tc.to(device),
            "gt_k_per_item": tk.to(device),
            "num_hits_per_item": tnh.to(device)
        })
    print(f"Pre-generated {len(pregen_train_data)} training batches.")

    print("\nPre-generating evaluation dataset...")
    eval_hit_matrix, eval_true_c_matrix, eval_true_k_per_item, eval_true_n_hits_per_item = generate_pseudo_data(
        max_true_clusters=data_params["max_true_clusters"],
        max_hits_per_batch_item=data_params["max_hits_per_batch_item"],
        avg_true_clusters=data_params["avg_true_clusters"],
        avg_hits_per_cluster=data_params["avg_hits_per_cluster"],
        batch_size=data_params["eval_batch_size"],
        cluster_pos_feature_dim=F_cluster_out_gt,
        hit_pos_feature_dim=F_in_hits,
        ensure_min_one_hit_per_cluster=data_params["ensure_min_one_hit_per_cluster"]
    )
    eval_data_dict = {
        "x_hits_input_device": eval_hit_matrix.to(device),
        "gt_c_matrix_device": eval_true_c_matrix.to(device),
        "gt_k_per_item_device": eval_true_k_per_item.to(device),
        "num_hits_per_item_device": eval_true_n_hits_per_item.to(device), # For use on device
        "cpu_hit_matrix": eval_hit_matrix,
        "cpu_true_c_matrix": eval_true_c_matrix,
        "cpu_true_k_per_item": eval_true_k_per_item,
        "cpu_true_n_hits_per_item": eval_true_n_hits_per_item
    }
    print(f"Evaluation dataset generated with {data_params['eval_batch_size']} items.")
    return pregen_train_data, eval_data_dict


def perform_training_step(model, optimizer, batch_data, loss_params, writer, global_step, gradient_clip_val):
    # ... (Unchanged from slot_attention(1).py)
    optimizer.zero_grad()
    pred_features, pred_logits = model(
        batch_data["x_hits_input"],
        num_hits_per_item=batch_data["num_hits_per_item"]
    )

    avg_feat_loss, avg_exist_loss, div_loss = compute_cluster_loss_geomloss(
        pred_features, pred_logits,
        batch_data["gt_c_matrix"], batch_data["gt_k_per_item"],
        nn.BCEWithLogitsLoss(reduction='sum'),
        lambda_diversity=loss_params["lambda_diversity"],
        geomloss_sigma_diversity=loss_params["geomloss_sigma_diversity_value"],
        epsilon_diversity_weight_norm=loss_params["epsilon_diversity_weight_norm_value"],
        cost_type=loss_params["cost_type_feature_loss"],
        geomloss_blur=loss_params["geomloss_blur_train"],
        geomloss_nits=loss_params["geomloss_nits_train"]
    )
    combined_loss = (loss_params["lambda_feat"] * avg_feat_loss +
                     loss_params["lambda_exist"] * avg_exist_loss +
                     loss_params["lambda_diversity"] * div_loss)

    train_losses = {
        "combined": combined_loss.item(),
        "feature": avg_feat_loss.item(),
        "existence": avg_exist_loss.item(),
        "diversity": div_loss.item()
    }

    combined_loss.backward()
    if gradient_clip_val > 0:
        total_norm_after_clipping = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        writer.add_scalar('GradientsMeta/TotalNorm_AfterClipping', total_norm_after_clipping.item(), global_step)
    optimizer.step()

    writer.add_scalar('Loss/Train/Combined', train_losses["combined"], global_step)
    writer.add_scalar('Loss/Train/Feature', train_losses["feature"], global_step)
    writer.add_scalar('Loss/Train/Existence', train_losses["existence"], global_step)
    writer.add_scalar('Loss/Train/Diversity', train_losses["diversity"], global_step)
    return train_losses

def _accumulate_item_eval_metrics(item_idx, eval_pred_features_item_cpu, eval_pred_probs_item_cpu,
                                 eval_data_dict, cost_matrix_item_hung_cpu,
                                 eval_specific_params, F_cluster_out_gt, writer, eval_global_step,
                                 accumulated_eval_metrics, device, model_k_max,
                                 eval_pred_features_item_device, eval_gt_c_matrix_item_device):
    # ... (Unchanged from slot_attention(1).py)
    k_gt_i = eval_data_dict["cpu_true_k_per_item"][item_idx].item()
    active_pred_mask_item = (eval_pred_probs_item_cpu > eval_specific_params["existence_threshold_eval"]).squeeze(-1)
    pred_k_i = active_pred_mask_item.sum().item()

    accumulated_eval_metrics["k_pairs_for_confusion_matrix"].append((k_gt_i, pred_k_i))
    accumulated_eval_metrics["total_items_for_k_acc"] += 1
    if pred_k_i == k_gt_i:
        accumulated_eval_metrics["correct_k_predictions"] += 1

    row_ind_np, col_ind_np = np.array([]), np.array([])
    feature_loss_fn_hungarian_impl = nn.L1Loss(reduction='sum')

    if k_gt_i > 0 and cost_matrix_item_hung_cpu.size > 0 : # Check if cost_matrix_item_hung_cpu is not empty
        row_ind_np, col_ind_np = linear_sum_assignment(cost_matrix_item_hung_cpu)

        if row_ind_np.size > 0:
            accumulated_eval_metrics["num_matched_pairs"] += len(row_ind_np)
            # Ensure indices are within bounds before accessing cost_matrix_item_hung_cpu
            valid_match_mask = (row_ind_np < cost_matrix_item_hung_cpu.shape[0]) & \
                               (col_ind_np < cost_matrix_item_hung_cpu.shape[1])
            valid_row_ind_np = row_ind_np[valid_match_mask]
            valid_col_ind_np = col_ind_np[valid_match_mask]

            if valid_row_ind_np.size > 0:
                accumulated_eval_metrics["total_matched_cost"] += cost_matrix_item_hung_cpu[valid_row_ind_np, valid_col_ind_np].sum()

                matched_pred_f_hung = eval_pred_features_item_device[torch.from_numpy(valid_row_ind_np).long().to(device)]
                matched_gt_f_hung = eval_gt_c_matrix_item_device[torch.from_numpy(valid_col_ind_np).long().to(device)]
                dist_hung = feature_loss_fn_hungarian_impl(matched_pred_f_hung, matched_gt_f_hung)
                accumulated_eval_metrics["total_feature_dist_matched"] += dist_hung.item()

            # Use original row_ind_np for slot stats, as they refer to predicted slots
            accumulated_eval_metrics["slot_match_counts"][row_ind_np[row_ind_np < model_k_max]] += 1 # Ensure indices are valid
            accumulated_eval_metrics["probs_of_matched_slots"].extend(eval_pred_probs_item_cpu[row_ind_np[row_ind_np < eval_pred_probs_item_cpu.shape[0]]].flatten().tolist())

            item_tp_exist = active_pred_mask_item[row_ind_np[row_ind_np < active_pred_mask_item.shape[0]]].sum().item()
            accumulated_eval_metrics["TP_exist"] += item_tp_exist
            accumulated_eval_metrics["FN_exist"] += (k_gt_i - item_tp_exist)

    item_tp_for_fp_calc = 0
    if k_gt_i > 0 and row_ind_np.size > 0:
         item_tp_for_fp_calc = active_pred_mask_item[row_ind_np[row_ind_np < active_pred_mask_item.shape[0]]].sum().item()
    accumulated_eval_metrics["FP_exist"] += (pred_k_i - item_tp_for_fp_calc)

    unmatched_active_mask = active_pred_mask_item.copy()
    if k_gt_i > 0 and row_ind_np.size > 0:
        unmatched_active_mask[row_ind_np[row_ind_np < unmatched_active_mask.shape[0]]] = False
    accumulated_eval_metrics["probs_of_unmatched_active_slots"].extend(
        eval_pred_probs_item_cpu[unmatched_active_mask].flatten().tolist()
    )

    if item_idx == 0 and F_cluster_out_gt >= 2:
        plot_matched_scatter_to_tensorboard(writer, "EvalVisuals_Hungarian", eval_global_step,
                                            eval_data_dict["cpu_hit_matrix"][item_idx, :eval_data_dict["cpu_true_n_hits_per_item"][item_idx]].numpy(),
                                            eval_data_dict["cpu_true_c_matrix"][item_idx, :k_gt_i].numpy(),
                                            eval_pred_features_item_cpu,
                                            eval_pred_probs_item_cpu,
                                            row_ind_np, col_ind_np, # these are from original assignment
                                            item_idx, F_cluster_out_gt, eval_specific_params["existence_threshold_eval"])

        plot_tsne_features_to_tensorboard(writer, "EvalVisuals_Hungarian", eval_global_step,
                                            eval_data_dict["cpu_true_c_matrix"][item_idx, :k_gt_i].numpy(),
                                            eval_pred_features_item_cpu,
                                            eval_pred_probs_item_cpu,
                                            item_idx, eval_specific_params["existence_threshold_eval"], F_cluster_out_gt)


def _log_aggregated_eval_metrics(accumulated_eval_metrics, writer, eval_global_step, model_k_max):
    if accumulated_eval_metrics["num_matched_pairs"] > 0:
        avg_matched_cost_h = accumulated_eval_metrics["total_matched_cost"] / accumulated_eval_metrics["num_matched_pairs"]
        avg_feat_dist_match_h = accumulated_eval_metrics["total_feature_dist_matched"] / accumulated_eval_metrics["num_matched_pairs"]
        writer.add_scalar('EvalMetrics_Hungarian/AvgMatchedCost', avg_matched_cost_h, eval_global_step)
        writer.add_scalar('EvalMetrics_Hungarian/AvgFeatureDist_MatchedPairs', avg_feat_dist_match_h, eval_global_step)

    if accumulated_eval_metrics["total_items_for_k_acc"] > 0:
        k_acc_h = accumulated_eval_metrics["correct_k_predictions"] / accumulated_eval_metrics["total_items_for_k_acc"]
        writer.add_scalar('EvalMetrics_Hungarian/K_Accuracy', k_acc_h, eval_global_step)

    k_pairs = accumulated_eval_metrics["k_pairs_for_confusion_matrix"]
    avg_pred_k_h = np.mean([p for _, p in k_pairs]) if k_pairs else 0
    avg_true_k_h = np.mean([t for t, _ in k_pairs]) if k_pairs else 0
    writer.add_scalar('EvalMetrics_Hungarian/AvgPredictedK', avg_pred_k_h, eval_global_step)
    writer.add_scalar('EvalMetrics_Hungarian/AvgTrueK', avg_true_k_h, eval_global_step)

    tp_e_h = accumulated_eval_metrics["TP_exist"]
    fp_e_h = accumulated_eval_metrics["FP_exist"]
    fn_e_h = accumulated_eval_metrics["FN_exist"]
    precision_e_h = tp_e_h / (tp_e_h + fp_e_h) if (tp_e_h + fp_e_h) > 0 else 0.0
    recall_e_h = tp_e_h / (tp_e_h + fn_e_h) if (tp_e_h + fn_e_h) > 0 else 0.0
    f1_e_h = 2 * (precision_e_h * recall_e_h) / (precision_e_h + recall_e_h) if (precision_e_h + recall_e_h) > 0 else 0.0
    writer.add_scalar('EvalMetrics_Hungarian/Existence_Precision', precision_e_h, eval_global_step)
    writer.add_scalar('EvalMetrics_Hungarian/Existence_Recall', recall_e_h, eval_global_step)
    writer.add_scalar('EvalMetrics_Hungarian/Existence_F1', f1_e_h, eval_global_step)

    plot_slot_usage_to_tensorboard(writer, "EvalVisuals_Hungarian", eval_global_step,
                                    accumulated_eval_metrics["slot_match_counts"].numpy(),
                                    model_k_max)
    plot_existence_prob_histograms_to_tensorboard(writer, "EvalVisuals_Hungarian", eval_global_step,
                                                    np.array(accumulated_eval_metrics["probs_of_matched_slots"]),
                                                    np.array(accumulated_eval_metrics["probs_of_unmatched_active_slots"]))
    plot_k_confusion_matrix_to_tensorboard(writer, "EvalVisuals_Hungarian", eval_global_step,
                                            k_pairs, model_k_max)


def perform_evaluation(model, eval_data_dict, loss_params, eval_specific_params,
                       writer, eval_global_step, F_cluster_out_gt, model_k_max, device):
    # ... (Unchanged from slot_attention(1).py)
    model.eval()
    with torch.no_grad():
        eval_pred_features, eval_pred_logits = model(eval_data_dict["x_hits_input_device"], eval_data_dict["num_hits_per_item_device"])

        eval_feat_loss, eval_exist_loss, eval_div_loss = compute_cluster_loss_geomloss(
            eval_pred_features, eval_pred_logits,
            eval_data_dict["gt_c_matrix_device"], eval_data_dict["gt_k_per_item_device"],
            nn.BCEWithLogitsLoss(reduction='sum'),
            lambda_diversity=loss_params["lambda_diversity"],
            geomloss_sigma_diversity=loss_params["geomloss_sigma_diversity_value"],
            epsilon_diversity_weight_norm=loss_params["epsilon_diversity_weight_norm_value"],
            cost_type=loss_params["cost_type_feature_loss"],
            geomloss_blur=loss_params["geomloss_blur_train"],
            geomloss_nits=loss_params["geomloss_nits_train"],
        )
        eval_combined_loss_val = (loss_params["lambda_feat"] * eval_feat_loss +
                                  loss_params["lambda_exist"] * eval_exist_loss +
                                  loss_params["lambda_diversity"] * eval_div_loss)

        eval_losses = {
            "combined": eval_combined_loss_val.item(),
            "feature": eval_feat_loss.item(),
            "existence": eval_exist_loss.item(),
            "diversity": eval_div_loss.item()
        }
        print(f"  EVAL Loss: {eval_losses['combined']:.4f} (F:{eval_losses['feature']:.4f}, E:{eval_losses['existence']:.4f}, D:{eval_losses['diversity']:.4f})")

        writer.add_scalar('Loss/Eval/Combined', eval_losses["combined"], eval_global_step)
        writer.add_scalar('Loss/Eval/Feature', eval_losses["feature"], eval_global_step)
        writer.add_scalar('Loss/Eval/Existence', eval_losses["existence"], eval_global_step)
        writer.add_scalar('Loss/Eval/Diversity', eval_losses["diversity"], eval_global_step)

        accumulated_eval_metrics = {
            "total_matched_cost": 0.0, "num_matched_pairs": 0,
            "total_feature_dist_matched": 0.0,
            "correct_k_predictions": 0, "total_items_for_k_acc": 0,
            "slot_match_counts": torch.zeros(model_k_max, dtype=torch.long, device="cpu"),
            "TP_exist": 0, "FP_exist": 0, "FN_exist": 0,
            "probs_of_matched_slots": [],
            "probs_of_unmatched_active_slots": [],
            "k_pairs_for_confusion_matrix": []
        }
        num_eval_items_processed = eval_data_dict["x_hits_input_device"].shape[0]

        cost_p_norm = 1 if eval_specific_params["cost_type_feature_loss_for_hungarian"] == 'L1' else 2
        cost_matrix_C_full_eval_hung = torch.cdist(eval_pred_features.detach(), eval_data_dict["gt_c_matrix_device"], p=cost_p_norm) # Detach pred_features
        if eval_specific_params["cost_type_feature_loss_for_hungarian"] == 'L2_sq':
            cost_matrix_C_full_eval_hung = cost_matrix_C_full_eval_hung.pow(2)

        for i in range(num_eval_items_processed):
            k_gt_i = eval_data_dict["cpu_true_k_per_item"][i].item()
            # Slice cost matrix for current item: (K_max_pred, k_gt_i)
            cost_m_item_hung_cpu = cost_matrix_C_full_eval_hung[i, :, :k_gt_i].cpu().numpy()

            _accumulate_item_eval_metrics(
                item_idx=i,
                eval_pred_features_item_cpu=eval_pred_features[i].cpu().numpy(),
                eval_pred_probs_item_cpu=torch.sigmoid(eval_pred_logits[i]).cpu().numpy(),
                eval_data_dict=eval_data_dict,
                cost_matrix_item_hung_cpu=cost_m_item_hung_cpu,
                eval_specific_params=eval_specific_params,
                F_cluster_out_gt=F_cluster_out_gt,
                writer=writer,
                eval_global_step=eval_global_step,
                accumulated_eval_metrics=accumulated_eval_metrics,
                device=device,
                model_k_max=model_k_max,
                eval_pred_features_item_device=eval_pred_features[i],
                eval_gt_c_matrix_item_device=eval_data_dict["gt_c_matrix_device"][i, :k_gt_i, :]
            )

        _log_aggregated_eval_metrics(accumulated_eval_metrics, writer, eval_global_step, model_k_max)

    model.train()
    return eval_losses


def plot_final_loss_summary(loss_plot_data, log_dir):
    # ... (Unchanged from slot_attention(1).py)
    if loss_plot_data["epochs"]:
        print("\n--- Plotting Training and Evaluation Losses (Final Summary) ---")
        plt.figure(figsize=(10, 5))
        plt.plot(loss_plot_data["epochs"], loss_plot_data["train_loss_combined"], label="Train Combined Loss", marker='.')
        plt.plot(loss_plot_data["epochs"], loss_plot_data["eval_loss_combined"], label="Eval Combined Loss", marker='.')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Combined Loss vs. Epoch")
        plt.legend()
        plt.grid(True)
        loss_plot_filename = f"{log_dir}/final_loss_summary.png"
        try:
            plt.savefig(loss_plot_filename)
            print(f"Final loss summary saved to {loss_plot_filename}")
        except Exception as e:
            print(f"Could not save {loss_plot_filename}: {e}")
        plt.close()


def run_training_pipeline(model, optimizer, pregen_train_data, eval_data_dict,
                          training_hyperparams, loss_params, eval_specific_params,
                          profiler_config, writer, log_dir, F_cluster_out_gt, model_k_max, device):
    # ... (Unchanged from slot_attention(1).py, but print statement added for VoxelTransformer)
    num_epochs = training_hyperparams["num_epochs"]
    num_batches_per_epoch = len(pregen_train_data)
    loss_plot_data = {"epochs": [], "train_loss_combined": [], "eval_loss_combined": []}
    prof = None

    print("\n--- Starting Training ---")
    if hasattr(model, 'use_voxel_transformer') and model.use_voxel_transformer:
        print("--- Training with VoxelTransformerClusterer for initial slot representations ---")
    elif hasattr(model, 'use_dbscan') and model.use_dbscan:
        print("--- Training with DBSCAN for initial slot representations ---")
    else:
        print("--- Training with default learnable slot queries ---")


    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        shuffled_indices = np.random.permutation(num_batches_per_epoch)
        latest_train_loss_combined = 0.0 # Store the loss of the last batch for plotting if epoch avg not used

        for batch_iter_idx in range(num_batches_per_epoch):
            actual_batch_idx_in_pregen = shuffled_indices[batch_iter_idx]
            global_step = epoch * num_batches_per_epoch + batch_iter_idx

            # Profiler logic from original
            is_profiling_batch = (profiler_config["enabled"] and epoch == 0 and
                                  batch_iter_idx < (profiler_config["warmup_steps"] + profiler_config["active_steps"]))
            if is_profiling_batch and batch_iter_idx == profiler_config["warmup_steps"]: # Start profiler after warmup
                if prof is None: # Initialize profiler only once
                    print(f"\n--- Starting PyTorch Profiler recording for {profiler_config['active_steps']} active steps ---")
                    prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if device.type == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
                        schedule=torch.profiler.schedule(wait=0, warmup=0, active=profiler_config["active_steps"], repeat=1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{log_dir}/profile_epoch0"),
                        record_shapes=True, profile_memory=True, with_stack=True
                    )
                    prof.start()

            current_batch_data = pregen_train_data[actual_batch_idx_in_pregen]
            train_losses = perform_training_step(model, optimizer, current_batch_data, loss_params,
                                                 writer, global_step, training_hyperparams["gradient_clip_val"])
            latest_train_loss_combined = train_losses["combined"]

            if prof is not None and is_profiling_batch : # Check if current batch is within active profiling window
                prof.step()
                if batch_iter_idx == (profiler_config["warmup_steps"] + profiler_config["active_steps"] - 1) : # Last active step
                    prof.stop()
                    print(f"--- PyTorch Profiler finished. Trace saved to {log_dir}/profile_epoch0 ---")
                    prof = None # Ensure profiler stops and saves

            if (batch_iter_idx + 1) % training_hyperparams["train_print_interval"] == 0 or batch_iter_idx == num_batches_per_epoch - 1:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_iter_idx+1}/{num_batches_per_epoch}], Train Loss: {train_losses['combined']:.4f} (F:{train_losses['feature']:.4f}, E:{train_losses['existence']:.4f}, D:{train_losses['diversity']:.4f})")

        if (epoch + 1) % training_hyperparams["eval_every_n_epochs"] == 0 or epoch == num_epochs - 1:
            eval_global_step = (epoch + 1) * num_batches_per_epoch - 1
            print(f"  Epoch [{epoch+1}/{num_epochs}] Starting EVAL...")
            eval_losses = perform_evaluation(model, eval_data_dict, loss_params, eval_specific_params,
                                             writer, eval_global_step, F_cluster_out_gt, model_k_max, device)
            loss_plot_data["epochs"].append(epoch + 1)
            loss_plot_data["train_loss_combined"].append(latest_train_loss_combined) 
            loss_plot_data["eval_loss_combined"].append(eval_losses["combined"])
            # log_gradients_to_tensorboard(writer, model, eval_global_step)


        epoch_duration = time.time() - epoch_start_time
        if (epoch + 1) % training_hyperparams["train_print_interval"] == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"        Epoch [{epoch+1}/{num_epochs}] Duration: {epoch_duration:.2f}s")

    print("--- Training Finished ---")
    if prof: # If training finished before profiler completed its cycle
        prof.stop()
        print("--- PyTorch Profiler stopped at end of training. Trace might be incomplete. ---")
    
    writer.flush()
    writer.close()
    plot_final_loss_summary(loss_plot_data, log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Slot Attention Clustering Experiment")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file (default: config.yaml)')
    args = parser.parse_args()

    (data_params_config, F_in_hits_config, F_cluster_out_gt_config, model_params_config,
        training_hyperparams_config, loss_params_config, eval_specific_params_config,
        profiler_config_main, log_dir_main, writer_main, device_main) = setup_experiment_configuration(args.config)

    # Add learning_rate to model_params_config for initialize_model_and_optimizer
    # as it's a top-level param in training_params from config
    model_params_config["learning_rate"] = training_hyperparams_config["learning_rate"]

    model_main, optimizer_main = initialize_model_and_optimizer(
        model_params_dict=model_params_config,
        device=device_main
    )

    pregen_train_data_main, eval_data_dict_main = pregenerate_data_sets(
        data_params=data_params_config,
        F_cluster_out_gt=F_cluster_out_gt_config,
        F_in_hits=F_in_hits_config,
        num_pregen_train_batches=training_hyperparams_config["num_pregenerated_training_batches"],
        device=device_main
    )

    run_training_pipeline(
        model=model_main,
        optimizer=optimizer_main,
        pregen_train_data=pregen_train_data_main,
        eval_data_dict=eval_data_dict_main,
        training_hyperparams=training_hyperparams_config,
        loss_params=loss_params_config,
        eval_specific_params=eval_specific_params_config,
        profiler_config=profiler_config_main,
        writer=writer_main,
        log_dir=log_dir_main,
        F_cluster_out_gt=F_cluster_out_gt_config,
        model_k_max=model_params_config["k_max"],
        device=device_main
    )


# In[ ]:
