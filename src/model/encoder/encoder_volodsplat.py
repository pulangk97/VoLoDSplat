from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import MinkowskiEngine as ME
import torch.nn.init as init

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians

from .common.gaussian_adapter_depth import GaussianAdapter_depth, GaussianAdapterCfg


from .encoder import Encoder
from .visualization.encoder_visualizer_volodsplat_cfg import EncoderVisualizerVoLoDSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead

from .common.me_fea import project_features_to_me, project_features_to_me_lod

from ...geometry.projection import get_world_rays
from .common.sparse_net import SparseGaussianHead, SparseGaussianLevelsHead, SparseHiUNetWithAttention, SparseUNetWithAttention
from .common.mink_resnet import  MultiScaleSparseHead

def print_mem(tag: str = ""):
    if not torch.cuda.is_available():
        print(f"[MEM] {tag} - no CUDA")
        return
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[MEM] {tag} | allocated={allocated:.1f} MB reserved={reserved:.1f} MB")



@dataclass
class EncoderVoLoDSplatCfg:
    name: Literal["volodsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerVoLoDSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str
    monodepth_weights_path: str

    # multi-view matching
    local_mv_match: int

    # voxel resolution
    voxel_resolution: float
    max_levels: int

    emb_level: bool


class EncoderVoLoDSplat(Encoder[EncoderVoLoDSplatCfg]):
    def __init__(self, cfg: EncoderVoLoDSplatCfg) -> None:
        super().__init__(cfg)
        self.max_levels = self.cfg.max_levels
        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            monodepth_weights_path=cfg.monodepth_weights_path,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample features to the original resolution
        model_configs = {
            'vitg': {'in_channels': 768, 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }

        self.feature_upsampler = DPTHead(**model_configs["vitg"],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs["vitg"]["features"]
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter_depth(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # conv regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)

        num_gaussian_parameters = self.gaussian_adapter.d_in + 3 + 1 

        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1

        # 3D Sparse UNet
        self.spare_net = SparseHiUNetWithAttention(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            num_blocks=3,
                            use_attention=False,
                            max_levels=self.max_levels,
                            voxel_resolution=self.cfg.voxel_resolution,
                            embed_level=self.cfg.emb_level,
                            )


        # Create Gaussian head
        self.gaussian_head = SparseGaussianLevelsHead(in_channels, num_gaussian_parameters)

        # max_levels = 4
        # self.max_levels = self.cfg.max_levels

    def _sparse_to_batched(self, features, coordinates, batch_size, return_mask=False):
       
        device = features.device
        _, c = features.shape

        batch_features_list = []
        batch_sizes = []
        max_voxels = 0

        for batch_idx in range(batch_size):
            mask = coordinates[:, 0] == batch_idx
            batch_feats = features[mask]  # [N_i, C]
            batch_features_list.append(batch_feats)
            batch_sizes.append(batch_feats.shape[0])
            max_voxels = max(max_voxels, batch_feats.shape[0])

        # Create padded tensor [b, 1, N_max, C]
        batched_features = torch.zeros(batch_size, 1, max_voxels, c, device=device)

        # Create valid data mask [b, 1, N_max]
        if return_mask:
            valid_mask = torch.zeros(batch_size, 1, max_voxels, dtype=torch.bool, device=device)

        for batch_idx, batch_feats in enumerate(batch_features_list):
            n_voxels = batch_feats.shape[0]
            batched_features[batch_idx, 0, :n_voxels, :] = batch_feats
            if return_mask:
                valid_mask[batch_idx, 0, :n_voxels] = True

        if return_mask:
            return batched_features, valid_mask
        return batched_features

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        ues_voxelnet: bool = True,
        max_levels: Optional[int] = None
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None


        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']
        
        depth = depth_preds[-1]

        voxel_resolution = self.cfg.voxel_resolution
        
        
        
        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)

            return {
                "gaussians": None,
                "depths": depths
            }, [0]

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )
        
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')
        
        
        # unet input [BV, C, H, W]  [6, 101, 256, 448]
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)
        # [BV, C, H, W]
        out = self.gaussian_regressor(concat)
        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]
        # [BV, C, H, W]   [6, 164, 256, 448]    
        out = torch.cat(concat, dim=1) 

        sparse_input, aggregated_points, counts = project_features_to_me_lod(
                context["intrinsics"],
                context["extrinsics"],
                out,
                depth=depth, 
                voxel_resolution=voxel_resolution,
                b=b, v=v, num_levels=max_levels
                )



        aggregated_points = aggregated_points

        sparse_out = self.spare_net(sparse_input)   # 3D Sparse UNet
        for i, sparse_out_single in enumerate(sparse_out):
            if torch.equal(sparse_out_single.C, sparse_input[i].C) and sparse_out_single.F.shape[1] == sparse_input[i].F.shape[1]:
                # Create new feature tensor
                new_features = sparse_out_single.F + sparse_input[i].F

                sparse_out[i] = ME.SparseTensor(
                    features=new_features,
                    coordinate_map_key=sparse_out_single.coordinate_map_key,
                    coordinate_manager=sparse_out_single.coordinate_manager
                )
            else:
                # Handle coordinate mismatch
                print("Warning: Input and output coordinates inconsistent, skipping residual connection")
                sparse_out[i] = sparse_out_single
    
        gaussians_list = self.gaussian_head(sparse_out)
 

        del sparse_out,sparse_input,new_features
        
        # [B, V, H*W, 1, 1]
        depths = rearrange(depth, "b v h w -> b v (h w) () ()")
        print("max depth:", depths.max(), "min depth:", depths.min())

        num_levels = max_levels
        # num_levels = 5
        levels = [i for i in range(num_levels+1)]



        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            
            intermediate_depth = depth_preds[0]
            # Get voxel_feature
            intermediate_voxel_feature, median_points_list, counts = project_features_to_me_lod(
                context["intrinsics"],
                context["extrinsics"],
                out,
                depth=intermediate_depth,
                voxel_resolution=voxel_resolution,
                b=b, v=v, num_levels=max_levels
                )

            intermediate_out = self.spare_net(intermediate_voxel_feature)   # 3D Sparse UNet

            # new_inter_features = intermediate_out.F + intermediate_voxel_feature[0].F

            # Create new SparseTensor
            for i, intermediate_out_single in enumerate(intermediate_out):
                new_inter_features = intermediate_out_single.F + intermediate_voxel_feature[i].F

                intermediate_out[i] = ME.SparseTensor(
                    features=new_inter_features,
                    coordinate_map_key=intermediate_voxel_feature[i].coordinate_map_key,
                    coordinate_manager=intermediate_voxel_feature[i].coordinate_manager
                )

            intermedian_out_with_residual = intermediate_out

            intermediate_gaussians_list = self.gaussian_head(intermedian_out_with_residual)

            # Release variables no longer needed
            del intermediate_voxel_feature,intermediate_out,intermedian_out_with_residual


        else:
            intermediate_gaussians_list = None

        gaussian_output_list = []
        while len(gaussians_list) > 0:
            gaussians = gaussians_list.pop(0)

            level = levels.pop(0)
            # print(f"Processing level {level} with {gaussians.F.shape[0]} gaussians")

            aggregated_points_single = aggregated_points.pop(0)

            # Support batch_size > 1: separate features by batch index
            gaussian_params, valid_mask = self._sparse_to_batched(gaussians.F, gaussians.C, b, return_mask=True)  # [b, 1, N_max, 38], [b, 1, N_max]
            

            # center_points = gaussians.C[...,1:].float() * voxel_resolution * (2**(num_levels-1-level)) + voxel_resolution * (2**(num_levels-1-level)) / 2.0  # [N_total, 3], convert to world coordinates
            
            batched_points = self._sparse_to_batched(aggregated_points_single, gaussians.C, b)  # [b, 1, N_max, 3]
            # batched_points = self._sparse_to_batched(center_points, gaussians.C, b)  # [b, 1, N_max, 3]

            opacity_raw = gaussian_params[..., :1]  # [b, 1, N_max, 1]
            opacity_raw = torch.where(
                valid_mask.unsqueeze(-1),  # [b, 1, N_max, 1]
                opacity_raw,
                torch.full_like(opacity_raw, -20.0)  # sigmoid(-20) ≈ 2e-9，
            )
            opacities = opacity_raw.sigmoid().unsqueeze(-1)  #[b, 1, N_max, 1, 1]
            raw_gaussians = gaussian_params[..., 1:]    #[b, 1, N_max, 37]
            raw_gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
            )

            try:
                # Convert raw_gaussians to gaussian parameters
                gaussians = self.gaussian_adapter.forward(
                    extrinsics = context["extrinsics"],
                    intrinsics = context["intrinsics"],
                    opacities = opacities,
                    raw_gaussians = rearrange(raw_gaussians,"b v r srf c -> b v r srf () c"),
                    input_images =rearrange(context["image"], "b v c h w -> (b v) c h w"),   
                    depth = depth,
                    coordinate = gaussians.C,
                    points = batched_points,
                    voxel_resolution = voxel_resolution,
                    level = level,
                    feature = out,
                    max_level=max_levels
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                raise

        
            gaussians = Gaussians(
                rearrange(
                    gaussians.means,   #[2, 1, 256000, 1, 1, 3]
                    "b v r srf spp xyz -> b (v r srf spp) xyz",   #[2, 256000, 3]
                ),
                rearrange(
                    gaussians.covariances,  #[2, 1, 256000, 1, 1, 3, 3]
                    "b v r srf spp i j -> b (v r srf spp) i j",  #[2, 256000, 3, 3]
                ),
                rearrange(
                    gaussians.harmonics, #[2, 1, 256000, 1, 1, 3, 9]
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",  #[2, 256000, 3, 9]
                ),
                rearrange(
                    gaussians.opacities,  #[2, 1, 256000, 1, 1]
                    "b v r srf spp -> b (v r srf spp)",  #[2, 256000]
                ),
                rearrange(torch.ones_like(gaussians.opacities, dtype=torch.int64, device=gaussians.opacities.device) * level,
                  "b v r srf spp -> b (v r srf spp)",  #[2, 256000]
                ),  #[2, 256000]
            )

            if self.cfg.supervise_intermediate_depth and intermediate_gaussians_list is not None and len(intermediate_gaussians_list) > 0:

                intermediate_gaussians = intermediate_gaussians_list.pop(0)
                median_points = median_points_list.pop(0)

                # Support batch_size > 1: separate features by batch index
                gaussian_params, inter_valid_mask = self._sparse_to_batched(intermediate_gaussians.F, intermediate_gaussians.C, b, return_mask=True)  # [b, 1, N_max, 38], [b, 1, N_max]
                
                # center_points_median = intermediate_gaussians.C[...,1:].float() * voxel_resolution * (2**(num_levels-1-level)) + voxel_resolution * (2**(num_levels-1-level)) / 2.0  # [N_total, 3], convert to world coordinates
                batched_median_points = self._sparse_to_batched(median_points, intermediate_gaussians.C, b)  # [b, 1, N_max, 3]
                # batched_median_points = self._sparse_to_batched(center_points_median, intermediate_gaussians.C, b)  # [b, 1, N_max, 3]

                # Separate opacity and other parameters
                # Key fix: set opacity to extremely small value at padding positions
                inter_opacity_raw = gaussian_params[..., :1]  # [b, 1, N_max, 1]
                inter_opacity_raw = torch.where(
                    inter_valid_mask.unsqueeze(-1),
                    inter_opacity_raw,
                    torch.full_like(inter_opacity_raw, -20.0)
                )
                intermediate_opacities = inter_opacity_raw.sigmoid().unsqueeze(-1)  #[b, 1, N_max, 1, 1]
                intermediate_raw_gaussians = gaussian_params[..., 1:]    #[b, 1, N_max, 37]
                intermediate_raw_gaussians = rearrange(
                intermediate_raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
                )


                # Convert raw_gaussians to gaussian parameters
                intermediate_gaussians = self.gaussian_adapter.forward(
                    extrinsics = context["extrinsics"],
                    intrinsics = context["intrinsics"],
                    opacities = intermediate_opacities,
                    raw_gaussians = rearrange(intermediate_raw_gaussians,"b v r srf c -> b v r srf () c"),
                    input_images =rearrange(context["image"], "b v c h w -> (b v) c h w"),  
                    depth = intermediate_depth,
                    coordinate = intermediate_gaussians.C,
                    points = batched_median_points,
                    voxel_resolution = voxel_resolution,
                    level = level,
                    feature = out,
                    max_level=max_levels
                )
            
                intermediate_gaussians = Gaussians(
                    rearrange(
                        intermediate_gaussians.means,   
                        "b v r srf spp xyz -> b (v r srf spp) xyz",   
                    ),
                    rearrange(
                        intermediate_gaussians.covariances,  
                        "b v r srf spp i j -> b (v r srf spp) i j", 
                    ),
                    rearrange(
                        intermediate_gaussians.harmonics, 
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",  
                    ),
                    rearrange(
                        intermediate_gaussians.opacities,  
                        "b v r srf spp -> b (v r srf spp)",  
                    ),
                    rearrange(torch.ones_like(intermediate_gaussians.opacities, dtype=torch.int64, device=intermediate_gaussians.opacities.device) * level,
                        "b v r srf spp -> b (v r srf spp)",  #[2, 256000]
                    ),  #[2, 256000]
            )

            else:
                intermediate_gaussians = None


            if self.cfg.return_depth:
                # return depth prediction for supervision
                # depths  = torch.cat(depth_preds, dim=0)
                output_depth = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                ).squeeze(-1).squeeze(-1)
                
                # print(depths.shape)  # [B, V, H, W]  [2, 6, 256, 448]
                if intermediate_gaussians is not None:
                    gaussian_output_list.append({
                        "gaussians": gaussians,
                        "depths": output_depth,
                        "intermediate_gaussians": intermediate_gaussians
                    })
                    # return {
                    #     "gaussians": gaussians,
                    #     "depths": depths,
                    #     "intermediate_gaussians": intermediate_gaussians
                    # }
                else:
                    gaussian_output_list.append({
                        "gaussians": gaussians,
                        "depths": output_depth,
                    })
                    # return {
                    #     "gaussians": gaussians,
                    #     "depths": depths,
                    # }


        concat_gaussians_means = torch.cat([g["gaussians"].means for g in gaussian_output_list], dim=1)
        concat_gaussians_covariances = torch.cat([g["gaussians"].covariances for g in gaussian_output_list], dim=1)
        concat_gaussians_harmonics = torch.cat([g["gaussians"].harmonics for g in gaussian_output_list], dim=1)
        concat_gaussians_opacities = torch.cat([g["gaussians"].opacities for g in gaussian_output_list],    dim=1)      
        concat_gaussians_levels = torch.cat([g["gaussians"].levels for g in gaussian_output_list], dim=1)
        concated_gaussians = Gaussians(
            means=concat_gaussians_means,
            covariances=concat_gaussians_covariances,
            harmonics=concat_gaussians_harmonics,
            opacities=concat_gaussians_opacities,
            levels=concat_gaussians_levels
        )

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            

            concat_gaussians_intermediate_means = torch.cat([g["intermediate_gaussians"].means for g in gaussian_output_list if "intermediate_gaussians" in g], dim=1)
            concat_gaussians_intermediate_covariances = torch.cat([g["intermediate_gaussians"].covariances for g in gaussian_output_list if "intermediate_gaussians" in g], dim=1)
            concat_gaussians_intermediate_harmonics = torch.cat([g["intermediate_gaussians"].harmonics for g in gaussian_output_list if "intermediate_gaussians" in g], dim=1)
            concat_gaussians_intermediate_opacities = torch.cat([g["intermediate_gaussians"].opacities for g in gaussian_output_list if "intermediate_gaussians" in g],    dim=1)    
            concat_gaussians_intermediate_levels = torch.cat([g["intermediate_gaussians"].levels for g in gaussian_output_list if "intermediate_gaussians" in g], dim=1)

            concated_intermediate_gaussians = Gaussians(
                means=concat_gaussians_intermediate_means,
                covariances=concat_gaussians_intermediate_covariances,
                harmonics=concat_gaussians_intermediate_harmonics,
                opacities=concat_gaussians_intermediate_opacities,
                levels=concat_gaussians_intermediate_levels
            )
            concated_output = {
                "gaussians": concated_gaussians,
                "depths": gaussian_output_list[0]["depths"],
                "intermediate_gaussians": concated_intermediate_gaussians
            }
        else:
            concated_output = {
                "gaussians": concated_gaussians,
                "depths": gaussian_output_list[0]["depths"],
            }

        mask_list = []
        mask_intermediate_list = []
        mask = torch.zeros(concated_gaussians.means.shape[1], dtype=torch.bool, device=concated_gaussians.means.device)
        mask_intermediate = torch.zeros(concated_intermediate_gaussians.means.shape[1], dtype=torch.bool, device=concated_intermediate_gaussians.means.device) if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1 else None
        start_idx = 0
        start_intermediate_idx = 0
        for g in gaussian_output_list:
            mask[start_idx:start_idx + g["gaussians"].means.shape[1]] = True
            start_idx += g["gaussians"].means.shape[1]
            mask_list.append(mask.clone().expand(concated_gaussians.means.shape[0], -1))


            if "intermediate_gaussians" in g:
                mask_intermediate[start_intermediate_idx:start_intermediate_idx + g["intermediate_gaussians"].means.shape[1]] = True
                start_intermediate_idx += g["intermediate_gaussians"].means.shape[1]
                mask_intermediate_list.append(mask_intermediate.clone().expand(concated_intermediate_gaussians.means.shape[0], -1))


        for m in mask_list:
            m.requires_grad = False

        for m in mask_intermediate_list:
            m.requires_grad = False


        return concated_output, gaussian_output_list, mask_list, mask_intermediate_list

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
