from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):

    view_rotation = extrinsics[:3, :3].inverse()
    # Apply the rotation to the means (Gaussian positions).
    means = einsum(view_rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = view_rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC band
    harmonics_view_invariant = harmonics[..., 0]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        torch.logit(opacities[..., None]).detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    

def save_gaussian_ply(gaussians, visualization_dump, example, save_path):

    v, _, h, w = example["context"]["image"].shape[1:]

    # Calculate expected number of gaussians (assuming spp=1)
    spp = 1  # samples per pixel
    expected_num_gaussians = v * h * w * spp
    current_num_gaussians = gaussians.means.shape[1]

    print(f"Expected gaussians: {expected_num_gaussians}, Current gaussians: {current_num_gaussians}")
    print(f"Visualization dump keys: {list(visualization_dump.keys()) if visualization_dump else 'None'}")
    print(f"Gaussians attributes: {[attr for attr in dir(gaussians) if not attr.startswith('_')]}")

    # Function to pad or truncate tensor to match expected size
    def pad_or_truncate_gaussian_tensor(tensor, target_size, default_value=0.0):
        """Pad with zeros or truncate tensor to match target size"""
        current_size = tensor.shape[1]  # assuming shape is (batch, num_gaussians, ...)

        if current_size == target_size:
            return tensor
        elif current_size < target_size:
            # Pad with default values
            pad_size = target_size - current_size
            if len(tensor.shape) == 3:  # (batch, num_gaussians, features)
                pad_shape = (tensor.shape[0], pad_size, tensor.shape[2])
            elif len(tensor.shape) == 4:  # (batch, num_gaussians, features, extra)
                pad_shape = (tensor.shape[0], pad_size, tensor.shape[2], tensor.shape[3])
            else:
                pad_shape = (tensor.shape[0], pad_size) + tensor.shape[2:]

            pad_tensor = torch.full(pad_shape, default_value,
                                   dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=1)
        else:
            # Truncate
            return tensor[:, :target_size]

    # Pad or truncate gaussians to match expected size
    # For means, pad with (0, 0, 0) - points at origin
    padded_means = pad_or_truncate_gaussian_tensor(gaussians.means, expected_num_gaussians, 0.0)

    # For harmonics, pad with small positive values to avoid completely black points
    padded_harmonics = pad_or_truncate_gaussian_tensor(gaussians.harmonics, expected_num_gaussians, 0.1)

    # For opacities, pad with very small values (almost transparent)
    padded_opacities = pad_or_truncate_gaussian_tensor(gaussians.opacities, expected_num_gaussians, 0.01)

    # Handle visualization_dump tensors with fallback to gaussians attributes
    # Check if visualization_dump has the required keys, otherwise extract from gaussians
    if visualization_dump is not None and "scales" in visualization_dump:
        scales_tensor = visualization_dump["scales"]
    elif hasattr(gaussians, 'scales'):
        scales_tensor = gaussians.scales
    else:
        # Create default scales if not available
        scales_tensor = torch.full((1, current_num_gaussians, 3), 0.01,
                                  dtype=torch.float32, device=gaussians.means.device)

    if visualization_dump is not None and "rotations" in visualization_dump:
        rotations_tensor = visualization_dump["rotations"]
    elif hasattr(gaussians, 'rotations'):
        rotations_tensor = gaussians.rotations
    else:
        # Create default rotations (identity quaternions)
        rotations_tensor = torch.zeros((1, current_num_gaussians, 4),
                                      dtype=torch.float32, device=gaussians.means.device)
        rotations_tensor[..., 3] = 1.0  # w component = 1 for identity quaternion

    # Now pad/truncate these tensors
    padded_scales = pad_or_truncate_gaussian_tensor(scales_tensor, expected_num_gaussians, 0.01)
    padded_rotations = pad_or_truncate_gaussian_tensor(rotations_tensor, expected_num_gaussians, 0.0)

    # Set the last component of rotation quaternion to 1 for identity rotation for padded points
    if expected_num_gaussians > current_num_gaussians:
        padded_rotations[0, current_num_gaussians:, 3] = 1.0  # w component = 1 for identity quaternion

    # Transform means into camera space.
    means = rearrange(
        padded_means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w, spp=spp
    )

    # Create a mask to filter the Gaussians. throw away Gaussians at the
    # borders, since they're generally of lower quality.
    mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    GAUSSIAN_TRIM = 8
    mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

    def trim(element):
        element = rearrange(
            element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w, spp=spp
        )
        return element[mask][None]

    # convert the rotations from camera space to world space as required
    cam_rotations = trim(padded_rotations)[0]
    c2w_mat = repeat(
        example["context"]["extrinsics"][0, :, :3, :3],
        "v a b -> h w spp v a b",
        h=h,
        w=w,
        spp=spp,
    )
    c2w_mat = c2w_mat[mask]  # apply trim

    cam_rotations_np = R.from_quat(
        cam_rotations.detach().cpu().numpy()
    ).as_matrix()
    world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
    world_rotations = R.from_matrix(world_mat).as_quat()
    world_rotations = torch.from_numpy(world_rotations).to(
        padded_scales
    )

    export_ply(
        example["context"]["extrinsics"][0, 0],
        trim(padded_means)[0],
        trim(padded_scales)[0],
        world_rotations,
        trim(padded_harmonics)[0],
        trim(padded_opacities)[0],
        save_path,
    )


