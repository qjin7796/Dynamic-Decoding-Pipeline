# Compare brain RDMs with model RDMs
# For ordinal valued models, compare using Spearman rho-a; 
# for continuous valued models, compare using cosine similarity.
# @ Qiuhan Jin, 20/09/2023


import numpy as np, os, json, pickle, torch
from datetime import datetime
from itertools import combinations
from scipy.stats import rankdata
from scipy.spatial.distance import squareform
from scipy.ndimage import generate_binary_structure, label
from statsmodels.stats.multitest import fdrcorrection
from nilearn import image


def get_tensor_residuals(target_rdms, control_rdms):
    """ Regress out a set of RDMs in control_model_rdms 
        from brain_rdms (voxelwise)
        or target_model_rdm (single model).

        brain_rdms: must be (n_voxels, 10296) shape.
        target_model_rdm: must be (10296,) shape.
        control_model_rdms: must be (n_models, 10296) shape.
    """
    # Standardize tensors
    target_rdms = (target_rdms - torch.mean(target_rdms, dim=-1, keepdim=True)) / (torch.std(target_rdms, dim=-1, keepdim=True) + 1e-12)
    control_rdms = (control_rdms - torch.mean(control_rdms, dim=-1, keepdim=True)) / (torch.std(control_rdms, dim=-1, keepdim=True) + 1e-12)
    # Check target_rdm(s) shape
    if len(target_rdms.shape) == 1:
        assert target_rdms.shape == (10296,), 'target_rdm must be (10296,) shape'
    elif len(target_rdms.shape) == 2:
        assert target_rdms.shape[1] == 10296, 'target_rdms must be 10296 long in dim 1'
    else:
        raise ValueError('target_rdms must be a 1D or 2D tensor')
    # Check control_rdm(s) shape
    assert len(control_rdms.shape) == 2, 'control_rdms must be 2D tensor'
    assert control_rdms.shape[1] == 10296, 'control_rdms must be 10296 long in dim 1'

    # Regress out control_rdms from target_rdm
    if len(target_rdms.shape) == 1:
        W_target = torch.linalg.lstsq(control_rdms.t(), target_rdms.unsqueeze(-1)).solution
        target_rdms_res = (
            target_rdms.unsqueeze(-1) - (control_rdms.t() @ W_target)
        ).view(target_rdms.shape)
        target_rdms_res = target_rdms_res.type(target_rdms.dtype)
    else:
        ## Regress out for each target rdm stacked in dim 0
        def func_regress_out(x):
            W = torch.linalg.lstsq(control_rdms.t(), x.unsqueeze(-1)).solution
            # print(x.size(), control_rdms.size(), W.size())
            x_res = (
                x.unsqueeze(-1) - (control_rdms.t() @ W)
            ).view(x.shape)
            x_res = x_res.type(x.dtype)
            return x_res
        
        target_rdms_res = torch.vmap(func_regress_out)(target_rdms)
    return target_rdms_res


def get_tensor_squareform(data, num_conds):
    """ Given a numpy array or a torch tensor, return its squareform.
        Can accept shape (n, num_cond_pairs) or (num_cond_pairs,).
        Output shape (n, num_conds, num_conds) or (num_conds, num_conds).
        Output float32 torch tensor on cpu.
    """
    assert isinstance(data, (np.ndarray, torch.Tensor)), 'data must be numpy array or torch tensor'
    if len(data.shape) == 1:
        # data is 1D
        if isinstance(data, torch.Tensor):
            data_squareform = squareform(data.cpu().numpy())
        else:
            data_squareform = squareform(data)
    elif len(data.shape) == 2:
        # data is 2D tensor
        data_squareform = np.zeros((data.shape[0], num_conds, num_conds))
        if isinstance(data, torch.Tensor):
            for i in range(data.shape[0]):
                data_squareform[i] = squareform(data[i].cpu().numpy())
        else:
            for i in range(data.shape[0]):
                data_squareform[i] = squareform(data[i])
    else:
        raise ValueError('data must be a 1D or 2D tensor')
    output_tensor = torch.from_numpy(data_squareform)
    output_tensor = output_tensor.type(torch.float32)
    return output_tensor


def compute_r(tensor_x_squareform, tensor_y_squareform, triu_indices, perm_indices=None):
    """ Compute Pearson correlation between two tensors (uppter tri of the array). 
        Tensor mean is removed before comparing.
        
        https://rsatoolbox.readthedocs.io/en/latest/comparing.html
    """
    triu_ind_row = tuple(triu_indices[0])
    triu_ind_col = tuple(triu_indices[1])
    tensor_y = tensor_y_squareform[triu_ind_row, triu_ind_col]
    if perm_indices is not None:
        tensor_x_squareform_permuted = tensor_x_squareform[:, perm_indices][:, :, perm_indices]  # (num_voxels, num_conds, num_conds) torch.float32
        tensor_x = tensor_x_squareform_permuted[:, triu_ind_row, triu_ind_col]
    else:
        tensor_x = tensor_x_squareform[:, triu_ind_row, triu_ind_col]
    # Extract upper triangle of tensor_y_squareform (tensor) to a vector (tensor)
    tensor_a = tensor_x - torch.mean(tensor_x, dim=-1, keepdim=True)
    tensor_b = tensor_y - torch.mean(tensor_y, dim=-1, keepdim=True)
    tensor_a_norm = torch.norm(tensor_a, dim=-1)
    tensor_b_norm = torch.norm(tensor_b, dim=-1)
    corr = torch.inner(tensor_a, tensor_b) / (tensor_a_norm * tensor_b_norm)
    return corr


# def compute_partial_r(tensor_x, tensor_y):
#     """ Compute partial Pearson correlation between two arrays of vectors.
#         tensor_x: brain rdms (num_voxels, cond_pair_length)
#         tensor_y: model rdms (num_models, cond_pair_length)
#         For each voxel RDM in tensor_x, put it with all model RDMs in tensor_y,
#         and compute pairwise partial correlation, i.e. compute pairwise correlation 
#         while controlling/regressing out/partialling out all the other RDMs.

#         Return: (num_voxels, num_models + 1, num_models + 1) array
#         For each voxel, the num_models + 1 by num_models + 1 matrix is the partial correlation matrix.
#         The first row/col is the voxel RDM, the rest rows/cols are model RDMs in tensor_y row order.
    
#     Reference:
#         https://www.cosmomvpa.org/matlab/cosmo_target_dsm_corr_measure.html?highlight=partial%20correlation
#         https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4681537/
#         https://www-sciencedirect-com.kuleuven.e-bronnen.be/science/article/pii/S1053811920302676
#     """
#     # Standardize tensor, tensor can be a 1D tensor or 2D (num_tensors, tensor_len) tensor
#     tensor_x = (tensor_x - torch.mean(tensor_x, dim=-1, keepdim=True)) / (torch.std(tensor_x, dim=-1, keepdim=True) + 1e-12)
#     tensor_y = (tensor_y - torch.mean(tensor_y, dim=-1, keepdim=True)) / (torch.std(tensor_y, dim=-1, keepdim=True) + 1e-12)
#     num_models = tensor_y.shape[0]
    
#     # Define functions to broadcast
#     def broadcast_partial_r(batch_x):
#         x = torch.cat((batch_x.unsqueeze(0), tensor_y), dim=0)  # cat each voxel tensor with all model tensor
#         x_cov = torch.cov(x)
#         x_cov_inv = torch.linalg.solve(x_cov, torch.eye(num_models+1, device=x_cov.device))
#         x_dia = torch.diag(torch.sqrt(1/torch.diag(x_cov_inv)))
#         x_partial_r_matrix = -1 * (x_dia @ x_cov_inv @ x_dia)
#         return x_partial_r_matrix[0, 1:]

#     # vmap broadcast_partial_r to every voxel tensor
#     all_voxels_partial_r = torch.vmap(broadcast_partial_r)(tensor_x)
#     # print(datetime.now(), 'all_voxels_partial_r.size(): ', all_voxels_partial_r.size())
#     return all_voxels_partial_r


# def test_partial_r(brain_rdms, model_rdms, permute_indices, triu_indices):
#     """ For each permutation index list, permute the brain rdms, compute pairwise partial correlation matrix.
#         Input:
#             brain_rdms: (num_voxels, num_conds, num_conds) square array, voxel t-value/z-score distance matrix
#             model_rdms: (num_mocels, num_conds, num_conds) square array, model distance matrix
#     """
#     # Permute brain_rdms (tensor)
#     brain_rdms_permuted = brain_rdms[:, permute_indices][:, :, permute_indices]  # (num_voxels, num_conds, num_conds) torch.float32
#     # Extract upper triangle of brain_rdms_permuted (tensor) to a vector (tensor)
#     triu_ind_row = tuple(triu_indices[0])
#     triu_ind_col = tuple(triu_indices[1])
#     brain_rdms_permuted_triu = brain_rdms_permuted[:, triu_ind_row, triu_ind_col]  # (num_voxels, num_cond_pairs) torch.float32
#     model_rdms_triu = model_rdms[:, triu_ind_row, triu_ind_col]  # (num_models, num_cond_pairs) torch.float32
#     all_voxels_partial_r = compute_partial_r(brain_rdms_permuted_triu, model_rdms_triu)  # (num_voxels, num_models) torch.float32
#     return all_voxels_partial_r


# def compute_normalized_cosine(tensor_x, tensor_y):
#     """ Compute cosine similarity between two vectors (torch tensors) or arrays of vectors (n * torch tensors).
#         Vector is standardized before computing such that the norm is 1.
#     Reference:
#         https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
#         For two unit vectors, cosine similarity is equal to the inner product.
#     """
#     # Standardize tensor, tensor can be a 1D tensor or 2D (num_tensors, tensor_len) tensor
#     tensor_a = (tensor_x - torch.mean(tensor_x, dim=-1, keepdim=True)) / (torch.std(tensor_x, dim=-1, keepdim=True) + 1e-12)
#     tensor_b = (tensor_y - torch.mean(tensor_y, dim=-1, keepdim=True)) / (torch.std(tensor_y, dim=-1, keepdim=True) + 1e-12)
#     # Compute tensor norm
#     tensor_b_norm = torch.norm(tensor_b, dim=-1)  # tensor_a and tensor_b have the same norm
#     # Compute cosine similarity
#     cos = torch.inner(tensor_a, tensor_b) / (tensor_b_norm ** 2)
#     return cos


# def test_normalized_cosine(brain_rdms, model_rdm, permute_indices, triu_indices):
#     """ For each permutation index list, permute the brain rdms, compute the cosine similarity with model rdm.
#         Input:
#             brain_rdms: (num_voxels, num_conds, num_conds) square array, voxel t-value/z-score distance matrix
#             model_rdm: (num_conds, num_conds) square array, model distance matrix
#     """
#     # # Check if brain_rdms is a (num_voxels, num_conds, num_conds) array
#     # assert len(brain_rdms.shape) == 3 and brain_rdms.shape[1] == brain_rdms.shape[2], 'brain_rdms must be a (num_voxels, num_conds, num_conds) array'
#     # # Check if model_rdm is a (num_conds, num_conds) array
#     # assert len(model_rdm.shape) == 2 and model_rdm.shape[0] == model_rdm.shape[1], 'model_rdm must be a (num_conds, num_conds) array'
#     # # Check if permute_indices is a tuple list of length = num_conds
#     # num_conds = model_rdm.shape[-1]
#     # assert len(permute_indices) == num_conds, 'permute_indices must be a tuple list of length = num_conds'
    
#     # Permute brain_rdms (tensor)
#     brain_rdms_permuted = brain_rdms[:, permute_indices][:, :, permute_indices]  # (num_voxels, num_conds, num_conds) torch.float32
#     # Extract upper triangle of brain_rdms_permuted (tensor) to a vector (tensor)
#     triu_ind_row = tuple(triu_indices[0])
#     triu_ind_col = tuple(triu_indices[1])
#     brain_rdms_permuted_triu = brain_rdms_permuted[:, triu_ind_row, triu_ind_col]  # (num_voxels, num_cond_pairs) torch.float32
#     model_rdm_triu = model_rdm[triu_ind_row, triu_ind_col]  # (num_cond_pairs,) torch.float32
#     cos = compute_normalized_cosine(brain_rdms_permuted_triu, model_rdm_triu)  # (num_voxels,) torch.float32
#     return cos


# def get_tensor_rank(data):
#     """ np.apply_along_axis(rankdata, -1, data), tied rank.
#         Output float32 torch tensor on cpu.
#     """
#     # Check input is numpy array or torch tensor
#     assert isinstance(data, (np.ndarray, torch.Tensor)), 'data must be numpy array or torch tensor'
#     if isinstance(data, torch.Tensor):
#         data_rank = np.apply_along_axis(rankdata, -1, data.cpu().numpy())
#     else:
#         data_rank = np.apply_along_axis(rankdata, -1, data)
#     output_tensor = torch.from_numpy(data_rank)
#     output_tensor = torch.type(output_tensor, torch.float32)
#     return output_tensor


# def compute_spearman_rhoa(rank_tensor_x_squareform, rank_tensor_y_squareform, perm_indices=None, triu_indices=None):
#     """ Compute Spearman rho-a between two rank tensors (uppter tri of the array). 
#         Tensor mean is removed before comparing.
        
#         https://rsatoolbox.readthedocs.io/en/latest/comparing.html
#         https://github.com/rsagroup/rsatoolbox/blob/main/src/rsatoolbox/rdm/compare.py#L185 

#         rank_tensor_x_squareform can be batched, i.e. (num_tensors, num_conds, num_conds) tensor.
#         rank_tensor_y_squareform must be a (num_conds, num_conds) tensor.
#         if perm_indices and triu_indices are given, permute rank_tensor_x before computing rho-a.
#     """
#     triu_ind_row = tuple(triu_indices[0])
#     triu_ind_col = tuple(triu_indices[1])
#     rank_tensor_y = rank_tensor_y_squareform[triu_ind_row, triu_ind_col]  # (num_cond_pairs,) torch.float32
#     if perm_indices is not None and triu_indices is not None:
#         # Permute rank_tensor_x_squareform (tensor)
#         rank_tensor_x_squareform_permuted = rank_tensor_x_squareform[:, perm_indices][:, :, perm_indices]  # (num_voxels, num_conds, num_conds) torch.float32
#         rank_tensor_x = rank_tensor_x_squareform_permuted[:, triu_ind_row, triu_ind_col]
#     else:
#         rank_tensor_x = rank_tensor_x_squareform[triu_ind_row, triu_ind_col]  # (num_cond_pairs,) torch.float32
#     # Centerize tensor, rank_tensor_x can be a 1D tensor or 2D (num_tensors, tensor_len) tensor
#     tensor_a = rank_tensor_x - torch.mean(rank_tensor_x, dim=-1, keepdim=True)  # func applied row-wise
#     tensor_b = rank_tensor_y - torch.mean(rank_tensor_y, dim=-1, keepdim=True)  # func applied row-wise
#     # Compute a b inner product
#     tensor_len = rank_tensor_x.shape[-1]
#     rho_a = 12 * torch.inner(tensor_a, tensor_b) / (tensor_len ** 3 - tensor_len)
#     return rho_a


# def permutation_test_kendall_tau_regout(brain_rdms, target_model_rdm, control_model_rdms, permute_indices, triu_indices):
#     # Permute brain_rdms (tensor)
#     brain_rdms_permuted = brain_rdms[:, permute_indices][:, :, permute_indices]  # (num_voxels, num_conds, num_conds) torch.float32
#     # Extract upper triangle of brain_rdms_permuted (tensor) to a vector (tensor)
#     triu_ind_row = tuple(triu_indices[0])
#     triu_ind_col = tuple(triu_indices[1])
#     brain_rdms_permuted_triu = brain_rdms_permuted[:, triu_ind_row, triu_ind_col]  # (num_voxels, num_cond_pairs) torch.float32
#     target_model_rdm_triu = target_model_rdm[:, triu_ind_row, triu_ind_col]  # (1, num_cond_pairs)
#     control_model_rdms_triu = control_model_rdms[:, triu_ind_row, triu_ind_col]  # (num_models, num_cond_pairs) torch.float32
#     # Compute kendall tau with control models regressed out
#     all_voxels_kendall_tau = get_kendall_tau_regout(brain_rdms_permuted_triu, target_model_rdm_triu, control_model_rdms_triu)
#     return all_voxels_kendall_tau


# def permutation_test_kendall_tau(brain_rdms, model_rdm, permute_indices, triu_indices):
#     """ For each permutation index list, permute the brain rdms, compute pairwise partial correlation matrix.
#         Input:
#             brain_rdms: (num_voxels, num_conds, num_conds) square array, voxel t-value/z-score distance matrix
#             model_rdms: (num_mocels, num_conds, num_conds) square array, model distance matrix
#     """
#     if permute_indices is not None:
#         # Permute brain_rdms (tensor)
#         brain_rdms_permuted = brain_rdms[:, permute_indices][:, :, permute_indices]  # (num_voxels, num_conds, num_conds) torch.float32
#         # Extract upper triangle of brain_rdms_permuted (tensor) to a vector (tensor)
#         triu_ind_row = tuple(triu_indices[0])
#         triu_ind_col = tuple(triu_indices[1])
#         brain_rdms_permuted_triu = brain_rdms_permuted[:, triu_ind_row, triu_ind_col]  # (num_voxels, num_cond_pairs) torch.float32
#         model_rdm_triu = model_rdm[:, triu_ind_row, triu_ind_col]  # (1, num_cond_pairs) torch.float32
#     else:
#         brain_rdms_permuted = brain_rdms[:, triu_ind_row, triu_ind_col]
#         model_rdm_triu = model_rdm[:, triu_ind_row, triu_ind_col]
    
#     def broadcast_kendall_tau(x):
#         return kendall(x.squeeze(), model_rdm_triu.squeeze())
    
#     all_voxels_kendall_tau = torch.vmap(broadcast_kendall_tau)(brain_rdms_permuted_triu)  # (num_voxels, num_models) torch.float32
#     return all_voxels_kendall_tau
