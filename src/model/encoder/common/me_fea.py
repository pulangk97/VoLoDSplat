import torch
from einops import rearrange, repeat
from ....geometry.projection import get_world_rays
from ....geometry.projection import sample_image_grid
import torch.nn.functional as F
import MinkowskiEngine as ME


from torch_scatter import scatter_mean, scatter_max, scatter_softmax
from src.utils.utils import save_point_cloud_color, save_point_cloud_color_PC, save_voxel_pc


def project_features_to_me(intrinsics, extrinsics, out, depth, voxel_resolution, b, v):
    device = out.device

    h, w = depth.shape[2:]
    _, c, _, _ = out.shape

    intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j")
    extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
    depths = rearrange(depth, "b v h w -> b v (h w) () ()")

    uv_grid = sample_image_grid((h, w), device)[0]
    uv_grid = repeat(uv_grid, "h w c -> 1 v (h w) () () c", v=v)
    origins, directions = get_world_rays(uv_grid, extrinsics, intrinsics)
    world_coords = origins + directions * depths[..., None]
    world_coords = world_coords.squeeze(3).squeeze(3)  # [B, V, N, 3]

    features = rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v)
    features = rearrange(features, "b v c h w -> b v h w c")
    features = rearrange(features, "b v h w c -> b v (h w) c")  # [B, V, N, C]

    all_points = rearrange(world_coords, "b v n c -> (b v n) c")  # [B*V*N, 3]
    feats_flat = features.reshape(-1, c)  # [B*V*N, C]


    with torch.no_grad():
        quantized_coords = torch.round(all_points / voxel_resolution).long()

        # Create coordinate matrix: batch index + quantized coordinates
        batch_indices = torch.arange(b, device=device).repeat_interleave(v * h * w).unsqueeze(1)
        combined_coords = torch.cat([batch_indices, quantized_coords], dim=1)

        # Get unique voxel IDs and mapping indices
        unique_coords, inverse_indices, counts = torch.unique(
            combined_coords,
            dim=0,
            return_inverse=True,
            return_counts=True
        )

    num_voxels = unique_coords.shape[0]

    aggregated_feats = torch.zeros(num_voxels, c, device=device)
    aggregated_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, c), feats_flat)
    aggregated_feats = aggregated_feats / counts.view(-1, 1).float()  # Average features

    aggregated_points = torch.zeros(num_voxels, 3, device=device)
    aggregated_points.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), all_points)
    aggregated_points = aggregated_points / counts.view(-1, 1).float()

    # Use correct coordinate format: batch index + quantized coordinates
    sparse_tensor = ME.SparseTensor(
        features=aggregated_feats,
        coordinates=unique_coords.int(),
        tensor_stride=1,
        device=device
    )

    return sparse_tensor, aggregated_points, counts






def project_features_to_me_lod(
    intrinsics, extrinsics, out, depth, voxel_resolution, b, v,
    num_levels=4, entropy_percentage=0.8, image = None, index = None, rays = None, keep_all_points_for_abl=False
): #0.8

    device = out.device

    h, w = depth.shape[2:]
    _, c, _, _ = out.shape

    intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j")
    extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
    depths = rearrange(depth, "b v h w -> b v (h w) () ()")

    if rays is None:
        uv_grid = sample_image_grid((h, w), device)[0]
        uv_grid = repeat(uv_grid, "h w c -> 1 v (h w) () () c", v=v)
        origins, directions = get_world_rays(uv_grid, extrinsics, intrinsics)
        world_coords = origins + directions * depths[..., None]
        world_coords = world_coords.squeeze(3).squeeze(3)  # [B, V, N, 3]
    else:
        origins = rearrange(rays["origins"], "b v h w c -> b v (h w) c")
        directions = rearrange(rays["directions"], "b v h w c -> b v (h w) c")
        ray_depths = rearrange(depth, "b v h w -> b v (h w) 1")
        world_coords = origins + directions * ray_depths  # [B, V, N, 3]




    features = rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v)
    features = rearrange(features, "b v c h w -> b v h w c")
    features = rearrange(features, "b v h w c -> b v (h w) c")  # [B, V, N, C]

    all_points = rearrange(world_coords, "b v n c -> (b v n) c")  # [B*V*N, 3]
    feats_flat = features.reshape(-1, c)  # [B*V*N, C]


    if image is not None and index is not None:
        import numpy as np
        points = []
        colors = []
        for vi in range(depth.shape[1]):
            valid_mask = depth[0, vi].reshape(-1) > 0
            points.append(world_coords[0, vi][valid_mask].detach().cpu().numpy())
            colors.append(
                image[0, vi]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 3)[valid_mask.detach().cpu().numpy()]
            )
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        save_point_cloud_color_PC(points, colors=colors, save_path=f"sampled_points_all.ply")
        # import numpy as np
        # from PIL import Image

        # Image.fromarray(np.uint8(image[0, 0, ...].permute(1, 2, 0).cpu().numpy() * 255.0)).save(
        #     f"sampled_points_image_{index}.png"
        # )
        # image_vis = rearrange(image, "b v c h w -> b v (h w) c")
        # save_point_cloud_color_PC(
        #     world_coords[0].reshape(-1, 3).cpu().numpy(),
        #     colors=image_vis[0].reshape(-1, 3).cpu().numpy(),
        #     save_path=f"sampled_points_{index}.ply",
        # )


    if image is not None:
        image = rearrange(image, "(b v) c h w -> b v c h w", b=b, v=v)
        image = rearrange(image, "b v c h w -> b v h w c")
        image = rearrange(image, "b v h w c -> b v (h w) c")  # [B, V, N, C]
        images_flat = image.reshape(-1, 3)  # [B*V*N, 3]



    all_sparse_tensors = []
    all_aggregated_points = []
    all_counts = []
    if image is not None:
        all_color_points = []

    # base_voxel_res = voxel_resolution * 16
    base_voxel_res = voxel_resolution * (2**num_levels)
    batch_indices = torch.arange(b, device=device).repeat_interleave(v * h * w).unsqueeze(1)

    for li in range(num_levels + 1):
        with torch.no_grad():
            cur_res = base_voxel_res / (2**li)


            quantized_coords = torch.round(all_points / cur_res).long()

            combined_coords = torch.cat([batch_indices, quantized_coords], dim=1)

            # unique_coords: [M, 4], inv_idx: [N_current], counts: [M]
            unique_coords, inverse_indices, counts = torch.unique(
                combined_coords, dim=0, return_inverse=True, return_counts=True
            )


        num_voxels = unique_coords.shape[0]



        aggregated_feats = torch.zeros(num_voxels, c, device=device)
        aggregated_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, c), feats_flat)
        aggregated_feats = aggregated_feats / counts.view(-1, 1).float()  # Average features


        if image is not None:
            aggregated_colors = torch.zeros(num_voxels, 3, device=device)
            aggregated_colors.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), images_flat)
            aggregated_colors = aggregated_colors / counts.view(-1, 1).float()  # Average colors

            sparse_color_tensor = ME.SparseTensor(
                features=aggregated_colors,
                coordinates=unique_coords.int(),
                tensor_stride=1,
                device=device
            )
            all_color_points.append(sparse_color_tensor)

        aggregated_points = torch.zeros(num_voxels, 3, device=device)
        aggregated_points.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), all_points)
        aggregated_points = aggregated_points / counts.view(-1, 1).float()

        # Use correct coordinate format: batch index + quantized coordinates
        sparse_tensor = ME.SparseTensor(
            features=aggregated_feats,
            coordinates=unique_coords.int(),
            tensor_stride=1,
            device=device
        )

        all_sparse_tensors.append(sparse_tensor)
        all_aggregated_points.append(aggregated_points)
        all_counts.append(counts)


        inv_idx = inverse_indices
        agg_feats = aggregated_feats
        with torch.no_grad():

            # agg_points = aggregated_points



            norm_feats = F.normalize(feats_flat, p=2, dim=1)
            norm_agg_feats = F.normalize(agg_feats, p=2, dim=1)


            dot_sim = (norm_feats * norm_agg_feats[inv_idx]).sum(dim=1)



            sim_probs = scatter_softmax(dot_sim / 0.1, inv_idx)



            p_log_p = sim_probs * torch.log(sim_probs + 1e-8)

            voxel_entropy = -scatter_mean(p_log_p, inv_idx, dim=0) * counts.float()



            # voxel_entropy += torch.randn_like(voxel_entropy) * 1e-6


            valid_voxel_mask = counts > 1  # [M]
            selected_voxel_mask = torch.zeros_like(valid_voxel_mask, dtype=torch.bool)

            if valid_voxel_mask.any():
                valid_entropy = voxel_entropy[valid_voxel_mask]
                valid_batch_ids = unique_coords[valid_voxel_mask, 0]


                selected_valid_mask = torch.zeros_like(valid_entropy, dtype=torch.bool)
                for i in range(b):
                    b_mask = (valid_batch_ids == i)
                    if not b_mask.any():
                        continue

                    b_entropy = valid_entropy[b_mask]
                    num_preserve = max(1, int(len(b_entropy) * entropy_percentage))
                    kth = max(1, len(b_entropy) - num_preserve + 1)
                    threshold = torch.kthvalue(b_entropy, kth).values
                    selected_valid_mask[b_mask] = b_entropy >= threshold

                selected_voxel_mask[valid_voxel_mask] = selected_valid_mask



        final_mask = selected_voxel_mask[inv_idx]

        # final_mask = entropy_mask[inv_idx]
        if keep_all_points_for_abl:
            final_mask = torch.ones_like(final_mask, dtype=torch.bool)

        all_points = all_points[final_mask]
        feats_flat = feats_flat[final_mask]
        batch_indices = batch_indices[final_mask]

        if image is not None:
            images_flat = images_flat[final_mask]

        if all_points.shape[0] == 0:
                break



        if index is not None:
            b0_mask = unique_coords[:, 0] == 0
            b0_coords = unique_coords[b0_mask, 1:4].float() * cur_res
            save_voxel_pc(
                b0_coords,
                counts[b0_mask].float(),
                f"lod_level_{index}_{li}_all_voxels.ply"
            )

            if b0_mask.any():
                save_voxel_pc(
                    b0_coords,
                    voxel_entropy[b0_mask],
                    f"lod_level_{index}_{li}_entropy_map.ply",
                    cmap="hot"
                )
    if image is not None:
        return all_sparse_tensors, all_aggregated_points, all_counts, all_color_points
    else:
        return all_sparse_tensors, all_aggregated_points, all_counts




# def project_features_to_me_lod(
#     intrinsics, extrinsics, out, depth, voxel_resolution, b, v,
#     num_levels=4, entropy_percentage=0.8, image = None, index = None
# ):

#     device = out.device

#     h, w = depth.shape[2:]
#     _, c, _, _ = out.shape

#     intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j")
#     extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
#     depths = rearrange(depth, "b v h w -> b v (h w) () ()")

#     uv_grid = sample_image_grid((h, w), device)[0]
#     uv_grid = repeat(uv_grid, "h w c -> 1 v (h w) () () c", v=v)
#     origins, directions = get_world_rays(uv_grid, extrinsics, intrinsics)
#     world_coords = origins + directions * depths[..., None]
#     world_coords = world_coords.squeeze(3).squeeze(3)  # [B, V, N, 3]

#     features = rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v)
#     features = rearrange(features, "b v c h w -> b v h w c")
#     features = rearrange(features, "b v h w c -> b v (h w) c")  # [B, V, N, C]

#     all_points = rearrange(world_coords, "b v n c -> (b v n) c")  # [B*V*N, 3]
#     feats_flat = features.reshape(-1, c)  # [B*V*N, C]

#     # device = out.device
#     # h, w = depth.shape[2:]
#     # _, c, _, _ = out.shape


#     # intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j")
#     # extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
#     # depths = rearrange(depth, "b v h w -> b v (h w) () ()")


#     # uv_grid = sample_image_grid((h, w), device)[0]
#     # uv_grid = repeat(uv_grid, "h w c -> 1 v (h w) () () c", v=v)
#     # origins, directions = get_world_rays(uv_grid, extrinsics, intrinsics)
#     # world_coords = origins + directions * depths[..., None]
#     # world_coords = world_coords.squeeze(3).squeeze(3)  # [B, V, N, 3]


#     # # uv_grid = ...
#     # # world_coords = ... (B, V, N, 3)


#     # features = rearrange(out, "(b v) c h w -> b v (h w) c", b=b, v=v)
#     # feats_flat = rearrange(features, "b v n c -> (b v n) c")
#     # all_points = rearrange(world_coords, "b v n c -> (b v n) c")


#     # import numpy as np
#     # from PIL import Image
#     # Image.fromarray(np.uint8(image[0,0,...].permute(1,2,0).cpu().numpy()*255.)).save(f'sampled_points_image_{index}.png')
#     # image = rearrange(image, "b v c h w -> b v (h w) c")
#     # save_point_cloud_color_PC(world_coords[0].reshape(-1,3).cpu().numpy(),colors=image[0].reshape(-1, 3).cpu().numpy(), save_path=f'sampled_points_{index}.ply')



#     # batch_indices = torch.arange(b, device=device).repeat_interleave(v * h * w)

#     all_sparse_tensors = []
#     all_aggregated_points = []
#     all_counts = []

#     # base_voxel_res = voxel_resolution * 16
#     base_voxel_res = voxel_resolution * 16
#     batch_indices = torch.arange(b, device=device).repeat_interleave(v * h * w).unsqueeze(1)
#     # with torch.no_grad():
#     for li in range(num_levels + 1):
#         with torch.no_grad():
#             cur_res = base_voxel_res / (2**li)


#             quantized_coords = torch.round(all_points / cur_res).long()

#             combined_coords = torch.cat([batch_indices, quantized_coords], dim=1)

#             # unique_coords: [M, 4], inv_idx: [N_current], counts: [M]
#             unique_coords, inverse_indices, counts = torch.unique(
#                 combined_coords, dim=0, return_inverse=True, return_counts=True
#             )



#             # agg_feats = scatter_mean(feats_flat, inv_idx, dim=0)
#             # agg_points = scatter_mean(all_points, inv_idx, dim=0)

#             # sparse_tensor = ME.SparseTensor(
#             #     features=agg_feats,
#             #     coordinates=unique_coords.int(),
#             #     device=device
#             # )

#         num_voxels = unique_coords.shape[0]

#         aggregated_feats = torch.zeros(num_voxels, c, device=device)
#         aggregated_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, c), feats_flat)
#         aggregated_feats = aggregated_feats / counts.view(-1, 1).float()  # Average features

#         aggregated_points = torch.zeros(num_voxels, 3, device=device)
#         aggregated_points.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), all_points)
#         aggregated_points = aggregated_points / counts.view(-1, 1).float()

#         # Use correct coordinate format: batch index + quantized coordinates
#         sparse_tensor = ME.SparseTensor(
#             features=aggregated_feats,
#             coordinates=unique_coords.int(),
#             tensor_stride=1,
#             device=device
#         )

#         all_sparse_tensors.append(sparse_tensor)
#         all_aggregated_points.append(aggregated_points)
#         all_counts.append(counts)


#         inv_idx = inverse_indices
#         agg_feats = aggregated_feats
#         with torch.no_grad():

#             # agg_points = aggregated_points



#             norm_feats = F.normalize(feats_flat, p=2, dim=1)
#             norm_agg_feats = F.normalize(agg_feats, p=2, dim=1)


#             dot_sim = (norm_feats * norm_agg_feats[inv_idx]).sum(dim=1)



#             sim_probs = scatter_softmax(dot_sim / 0.1, inv_idx)



#             p_log_p = sim_probs * torch.log(sim_probs + 1e-8)





#             voxel_entropy += torch.randn_like(voxel_entropy) * 1e-6

#             entropy_mask = torch.zeros(len(voxel_entropy), dtype=torch.bool, device=device)
#             for i in range(b):
#                 b_mask = (unique_coords[:, 0] == i)
#                 # torch.cuda.synchronize()
#                 if not b_mask.any():
#                     continue

#                 b_entropy = voxel_entropy[b_mask]
#                 num_preserve = max(1, int(len(b_entropy) * entropy_percentage))


#                 threshold = torch.kthvalue(b_entropy, max(1, len(b_entropy) - num_preserve + 1)).values
#                 entropy_mask[b_mask] = (b_entropy >= threshold)



#         pre_mask = (counts > 1)[inv_idx]
#         final_mask = pre_mask & entropy_mask[inv_idx]

#         # final_mask = entropy_mask[inv_idx]

#         all_points = all_points[final_mask]
#         feats_flat = feats_flat[final_mask]
#         batch_indices = batch_indices[final_mask]

#         if all_points.shape[0] == 0:
#                 break



#             ####### entropy visualization ##########
#             # b0_mask = (unique_coords[:, 0] == 0)



#             # save_voxel_pc(
#             #     b0_coords,
#             #     counts[b0_mask].float(),
#             #     f'lod_level_{index}_{li}_all_voxels.ply'
#             # )





#             # if b0_mask.any():
#             #     save_voxel_pc(
#             #         b0_coords,
#             #         voxel_entropy[b0_mask],
#             #         f'lod_level_{index}_{li}_entropy_map.ply',

#             #     )




#             # selected_b0 = b0_mask & entropy_mask
#             # if selected_b0.any():
#             #     save_voxel_pc(
#             #         unique_coords[selected_b0, 1:4].float() * cur_res,
#             #         voxel_entropy[selected_b0],
#             #         f'lod_level_{index}_{li}_selected_top_{entropy_percentage}.ply'
#             #     )


#     return all_sparse_tensors, all_aggregated_points, all_counts

