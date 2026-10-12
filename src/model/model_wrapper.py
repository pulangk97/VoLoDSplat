from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb

from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger


from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import math

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.vis_depth import viz_depth_tensor
from PIL import Image
from ..misc.stablize_camera import render_stabilization_path
from .ply_export import save_gaussian_ply
from .encoder.encoder_volodsplat import print_mem
import MinkowskiEngine as ME
from .types import Gaussians

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    lr_level_embedding: float
    weight_decay: float


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    l1_loss: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    viz_depth: bool
    forward_depth_only: bool
    train_ignore_large_loss: float
    no_log_projections: bool
    render_chunk_size: int | None


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        self.max_levels = self.encoder.max_levels

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

            # Add statistics tracking for memory, inference time, and Gaussian points
            self.inference_stats = {
                'memory_allocated': [],
                'memory_reserved': [],
                'encoder_mem_mb': [],
                'gaussians_mem_mb': [],
                'inference_times': [],
                'gaussian_counts': [],
                'step_count': 0
            }

    def _decoder_forward_chunked(
        self,
        gaussians: Gaussians,
        extrinsics: Tensor,
        intrinsics: Tensor,
        near: Tensor,
        far: Tensor,
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        chunk_size = self.train_cfg.render_chunk_size
        num_views = int(extrinsics.shape[1])

        if chunk_size is None or chunk_size <= 0 or num_views <= chunk_size:
            return self.decoder.forward(
                gaussians,
                extrinsics,
                intrinsics,
                near,
                far,
                image_shape,
                depth_mode=depth_mode,
            )

        output: DecoderOutput | None = None
        for start in range(0, num_views, chunk_size):
            end = min(start + chunk_size, num_views)
            curr_output = self.decoder.forward(
                gaussians,
                extrinsics[:, start:end],
                intrinsics[:, start:end],
                near[:, start:end],
                far[:, start:end],
                image_shape,
                depth_mode=depth_mode,
            )

            if output is None:
                output = curr_output
            else:
                output.color = torch.cat((output.color, curr_output.color), dim=1)
                if output.depth is not None and curr_output.depth is not None:
                    output.depth = torch.cat((output.depth, curr_output.depth), dim=1)

        assert output is not None
        return output

    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        _, views, _, _, _ = batch["context"]["image"].shape

        print(f"Training step{self.global_step},Number of images:{views}:scene IDs:{batch['scene']}")

        # Run the model.
        gaussians, gaussian_list, mask_list, mask_intermediate_list = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"], max_levels=self.max_levels
        )

        gaussian_count_concat = (
            gaussians["gaussians"].means.shape[1]
            if len(gaussians["gaussians"].means.shape) > 1
            else gaussians["gaussians"].means.shape[0]
        )
        gaussian_count_intermediate = 0
        if "intermediate_gaussians" in gaussians:
            gaussian_count_intermediate = (
                gaussians["intermediate_gaussians"].means.shape[1]
                if len(gaussians["intermediate_gaussians"].means.shape) > 1
                else gaussians["intermediate_gaussians"].means.shape[0]
            )


        multiscale_gaussians = []
        for i, m in enumerate(mask_list):

            single_scale_gaussian = Gaussians(
            means=gaussians["gaussians"].means[m].expand(gaussians["gaussians"].means.shape[0], -1, -1) if len(gaussians["gaussians"].means.shape) > 1 else gaussians["gaussians"].means[m].expand(gaussians["gaussians"].means.shape[0], -1),
            covariances=gaussians["gaussians"].covariances[m].expand(gaussians["gaussians"].covariances.shape[0], -1, -1, -1) if len(gaussians["gaussians"].covariances.shape) > 1 else gaussians["gaussians"].covariances[m].expand(gaussians["gaussians"].covariances.shape[0], -1, -1),
            harmonics=gaussians["gaussians"].harmonics[m].expand(gaussians["gaussians"].harmonics.shape[0], -1, -1, -1) if len(gaussians["gaussians"].harmonics.shape) > 1 else gaussians["gaussians"].harmonics[m].expand(gaussians["gaussians"].harmonics.shape[0], -1, -1),
            opacities=gaussians["gaussians"].opacities[m].expand(gaussians["gaussians"].opacities.shape[0], -1) if len(gaussians["gaussians"].opacities.shape) > 1 else gaussians["gaussians"].opacities[m].expand(gaussians["gaussians"].opacities.shape[0], -1),
            levels=gaussians["gaussians"].levels[m].expand(gaussians["gaussians"].levels.shape[0], -1) if len(gaussians["gaussians"].levels.shape) > 1 else gaussians["gaussians"].levels[m].expand(gaussians["gaussians"].levels.shape[0], -1)
            )

            if len(mask_intermediate_list)>0 and i < len(mask_intermediate_list):
                single_scale_gaussian_intermediate = Gaussians(
                    means=gaussians["intermediate_gaussians"].means[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].means.shape[0], -1, -1) if len(gaussians["intermediate_gaussians"].means.shape) > 1 else gaussians["intermediate_gaussians"].means[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].means.shape[0], -1),
                    covariances=gaussians["intermediate_gaussians"].covariances[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].covariances.shape[0], -1, -1, -1) if len(gaussians["intermediate_gaussians"].covariances.shape) > 1 else gaussians["intermediate_gaussians"].covariances[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].covariances.shape[0], -1, -1),
                    harmonics=gaussians["intermediate_gaussians"].harmonics[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].harmonics.shape[0], -1, -1, -1) if len(gaussians["intermediate_gaussians"].harmonics.shape) > 1 else gaussians["intermediate_gaussians"].harmonics[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].harmonics.shape[0], -1, -1),
                    opacities=gaussians["intermediate_gaussians"].opacities[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].opacities.shape[0], -1) if len(gaussians["intermediate_gaussians"].opacities.shape) > 1 else gaussians["intermediate_gaussians"].opacities[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].opacities.shape[0], -1),
                    levels=gaussians["intermediate_gaussians"].levels[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].levels.shape[0], -1) if len(gaussians["intermediate_gaussians"].levels.shape) > 1 else gaussians["intermediate_gaussians"].levels[mask_intermediate_list[i]].expand(gaussians["intermediate_gaussians"].levels.shape[0], -1)
                )
                multiscale_gaussians.append({"gaussians": single_scale_gaussian, "depths": gaussians["depths"], "intermediate_gaussians": single_scale_gaussian_intermediate})
            else:
                multiscale_gaussians.append({"gaussians": single_scale_gaussian, "depths": gaussians["depths"]})

        cumulated_gaussians = multiscale_gaussians
        total_loss = 0
        level = 0

        while len(cumulated_gaussians)>0:
            gaussians = cumulated_gaussians.pop(0)


            if isinstance(gaussians, dict) and len(gaussians) == 2:
                supervise_intermediate_depth = False
                pred_depths = gaussians["depths"]
                gaussians = gaussians["gaussians"]

            if isinstance(gaussians, dict) and len(gaussians) == 3:
                supervise_intermediate_depth = True
                pred_depths = gaussians["depths"]
                intermediate_gaussians = gaussians["intermediate_gaussians"]
                gaussians = gaussians["gaussians"]


            if gaussians.means.size(0) != batch["target"]["extrinsics"].size(0):
                supervise_intermediate_depth = True
                assert gaussians.means.size(0) % batch["target"]["extrinsics"].size(0) == 0
                num_depths = gaussians.means.size(0) // batch["target"]["extrinsics"].size(
                    0
                )
                # add loss to intermediate depth predictions
                target_extrinsics = torch.cat(
                    [batch["target"]["extrinsics"]] * num_depths, dim=0
                )
                target_intrinsics = torch.cat(
                    [batch["target"]["intrinsics"]] * num_depths, dim=0
                )
                target_near = torch.cat([batch["target"]["near"]] * num_depths, dim=0)
                target_far = torch.cat([batch["target"]["far"]] * num_depths, dim=0)
                #[2B, V, 3, 256, 448]
                output_all = self._decoder_forward_chunked(
                    gaussians,
                    target_extrinsics,
                    target_intrinsics,
                    target_near,
                    target_far,
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                )


                # split
                batch_size = batch["target"]["extrinsics"].size(0)
                # order: intermediate depth, final depth
                output_intermediate = DecoderOutput(
                    color=output_all.color[:-batch_size],   #[B, V, 3, H, W]
                    depth=(
                        output_all.depth[:-batch_size]
                        if output_all.depth is not None
                        else None
                    ),
                )
                output = DecoderOutput(
                    color=output_all.color[-batch_size:],   #[B, V, 3, H, W]
                    depth=(
                        output_all.depth[-batch_size:]
                        if output_all.depth is not None
                        else None
                    ),
                )

            else:
                output = self._decoder_forward_chunked(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                )
                if supervise_intermediate_depth:
                    output_intermediate = self._decoder_forward_chunked(
                        intermediate_gaussians,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        depth_mode=self.train_cfg.depth_mode,
                    )

            # print_mem("after decoder")
            target_gt = batch["target"]["image"]

            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )

            self.log(f"train/psnr_level{level}", psnr_probabilistic.mean())

            # Compute and log loss.
            # total_loss = 0

            valid_depth_mask = None

            for loss_fn in self.losses:
                if loss_fn.name == "mse":
                    loss = loss_fn.forward(
                        output,
                        batch,
                        gaussians,
                        self.global_step,
                        l1_loss=self.train_cfg.l1_loss,
                        clamp_large_error=self.train_cfg.train_ignore_large_loss,
                        valid_depth_mask=valid_depth_mask,
                        target_image=target_gt,
                    )
                else:
                    loss = loss_fn.forward(
                        output,
                        batch,
                        gaussians,
                        self.global_step,
                        valid_depth_mask=valid_depth_mask,
                        target_image=target_gt,
                    )
                self.log(f"loss/{loss_fn.name}_level{level}", loss)
                total_loss = total_loss + loss

            # color loss on intermediate output
            if supervise_intermediate_depth:
                for loss_fn in self.losses:
                    batch_size = batch["target"]["extrinsics"].size(0)
                    if output_intermediate.color.size(0) != batch_size:
                        assert output_intermediate.color.size(0) % batch_size == 0
                        num_intermediate = output_intermediate.color.size(0) // batch_size
                        intermediate_loss = 0
                        for i in range(num_intermediate):
                            curr_output = DecoderOutput(
                                color=output_intermediate.color[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ],
                                depth=(
                                    output_intermediate.depth[
                                        (batch_size * i) : (batch_size * (i + 1))
                                    ]
                                    if output_intermediate.depth is not None
                                    else None
                                ),
                            )
                            curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                                num_intermediate - i
                            )

                            if loss_fn.name == "mse":
                                loss = loss_fn.forward(
                                    curr_output,
                                    batch,
                                    gaussians,
                                    self.global_step,
                                    l1_loss=self.train_cfg.l1_loss,
                                    clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                    valid_depth_mask=valid_depth_mask,
                                    target_image=target_gt,
                                )
                            else:
                                loss = loss_fn.forward(
                                    curr_output,
                                    batch,
                                    gaussians,
                                    self.global_step,
                                    valid_depth_mask=valid_depth_mask,
                                    target_image=target_gt,
                                )

                            intermediate_loss = intermediate_loss + curr_loss_weight * loss

                        self.log(f"loss/{loss_fn.name}_intermediate_level{level}", intermediate_loss)
                        total_loss = total_loss + intermediate_loss
                    else:
                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                output_intermediate,
                                batch,
                                gaussians,
                                self.global_step,
                                l1_loss=self.train_cfg.l1_loss,
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                                target_image=target_gt,
                            )
                        else:
                            loss = loss_fn.forward(
                                output_intermediate,
                                batch,
                                gaussians,
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                                target_image=target_gt,
                            )
                        self.log(f"loss/{loss_fn.name}_intermediate_level{level}", loss)
                        total_loss = (
                            total_loss + self.train_cfg.intermediate_loss_weight * loss
                        )
            level+=1


        total_loss = total_loss / level if level > 0 else total_loss
        self.log("loss/total_all_level", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            mem_msg = ""
            if torch.cuda.is_available():
                mem_alloc_mb = torch.cuda.memory_allocated() / 1024**2
                mem_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
                mem_msg = f"; mem_alloc_mb = {mem_alloc_mb:.1f}; mem_peak_mb = {mem_peak_mb:.1f}"
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"gauss_concat = {gaussian_count_concat}; "
                f"gauss_intermediate = {gaussian_count_intermediate}; "
                f"loss = {total_loss:.6f}"
                f"{mem_msg}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor
        self.log("info/gaussian_count_concat", float(gaussian_count_concat))
        self.log("info/gaussian_count_intermediate", float(gaussian_count_intermediate))

        if torch.cuda.is_available():
            mem_alloc_mb = torch.cuda.memory_allocated() / 1024**2
            mem_reserved_mb = torch.cuda.memory_reserved() / 1024**2
            mem_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            self.log("info/gpu_mem_alloc_mb", float(mem_alloc_mb))
            self.log("info/gpu_mem_reserved_mb", float(mem_reserved_mb))
            self.log("info/gpu_mem_peak_mb", float(mem_peak_mb))

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")



        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        print("target_views:", v)
        _, v_c, _, _, _ = batch["context"]["image"].shape
        print("context_views:", v_c)
        assert b == 1

        inference_start_time = time.time() if self.test_cfg.compute_scores else None
        pred_depths = None
        decoder_total_time = 0.0
        decoder_time_concat = 0.0
        decoder_time_level = []
        level_gaussian_counts = []
        output_per_level = []

        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        (scene,) = batch["scene"]
        print(scene)
        # assert False, print("debug")
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        encoder_start_time = time.time() if self.test_cfg.compute_scores else None
        encoder_mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, gaussian_list, mask_list, _ = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
                max_levels=self.max_levels
            )

        encoder_mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        encoder_mem_mb = (
            max(0, encoder_mem_after - encoder_mem_before) / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )

        multiscale_gaussians = []
        for i, m in enumerate(mask_list):

            single_scale_gaussian = Gaussians(
            means=gaussians["gaussians"].means[m].expand(gaussians["gaussians"].means.shape[0], -1, -1) if len(gaussians["gaussians"].means.shape) > 1 else gaussians["gaussians"].means[m].expand(gaussians["gaussians"].means.shape[0], -1),
            covariances=gaussians["gaussians"].covariances[m].expand(gaussians["gaussians"].covariances.shape[0], -1, -1, -1) if len(gaussians["gaussians"].covariances.shape) > 1 else gaussians["gaussians"].covariances[m].expand(gaussians["gaussians"].covariances.shape[0], -1, -1),
            harmonics=gaussians["gaussians"].harmonics[m].expand(gaussians["gaussians"].harmonics.shape[0], -1, -1, -1) if len(gaussians["gaussians"].harmonics.shape) > 1 else gaussians["gaussians"].harmonics[m].expand(gaussians["gaussians"].harmonics.shape[0], -1, -1),
            opacities=gaussians["gaussians"].opacities[m].expand(gaussians["gaussians"].opacities.shape[0], -1) if len(gaussians["gaussians"].opacities.shape) > 1 else gaussians["gaussians"].opacities[m].expand(gaussians["gaussians"].opacities.shape[0], -1),
            levels=gaussians["gaussians"].levels[m].expand(gaussians["gaussians"].levels.shape[0], -1) if len(gaussians["gaussians"].levels.shape) > 1 else gaussians["gaussians"].levels[m].expand(gaussians["gaussians"].levels.shape[0], -1)
            )
            multiscale_gaussians.append({"gaussians": single_scale_gaussian, "depths": gaussians["depths"]})
        if len(multiscale_gaussians) == 0:
            multiscale_gaussians = [gaussians]

        max_levels = self.max_levels

        if self.test_cfg.compute_scores:
            encoder_time = time.time() - encoder_start_time if encoder_start_time is not None else 0
            if 'encoder_times' not in self.inference_stats:
                self.inference_stats['encoder_times'] = []
            self.inference_stats['encoder_times'].append(encoder_time)

            if torch.cuda.is_available():
                mem_after_encoder = torch.cuda.memory_allocated()
                mem_reserved = torch.cuda.memory_reserved()
                self.inference_stats['memory_allocated'].append(mem_after_encoder / 1024**2)  # MB
                self.inference_stats['memory_reserved'].append(mem_reserved / 1024**2)  # MB
                self.inference_stats.setdefault('encoder_mem_mb', []).append(encoder_mem_mb)
        # max_levels = self.max_levels
        level_gaussians_mem_mb = []
        for level in range(max_levels+1):
            # level = min(li, len(multiscale_gaussians)-1)

            level_gaussians = multiscale_gaussians[min(level, len(multiscale_gaussians)-1)]
        # for level, level_gaussians in enumerate(multiscale_gaussians):
            gaussians_for_level = level_gaussians

            if isinstance(gaussians_for_level, dict):
                pred_depths = gaussians_for_level.get("depths", pred_depths)
                if "depth" in batch["context"]:
                    depth_gt = batch["context"]["depth"]
                gaussians_for_level = gaussians_for_level["gaussians"]

            if hasattr(gaussians_for_level, "means"):
                gaussian_count = (
                    gaussians_for_level.means.shape[1]
                    if len(gaussians_for_level.means.shape) > 1
                    else gaussians_for_level.means.shape[0]
                )
                gaussians_mem_mb = (
                    gaussians_for_level.means.numel() * gaussians_for_level.means.element_size()
                    + gaussians_for_level.covariances.numel() * gaussians_for_level.covariances.element_size()
                    + gaussians_for_level.harmonics.numel() * gaussians_for_level.harmonics.element_size()
                    + gaussians_for_level.opacities.numel() * gaussians_for_level.opacities.element_size()
                    + gaussians_for_level.levels.numel() * gaussians_for_level.levels.element_size()
                ) / 1024**2
            elif hasattr(gaussians_for_level, "shape"):
                gaussian_count = (
                    gaussians_for_level.shape[1]
                    if len(gaussians_for_level.shape) > 1
                    else gaussians_for_level.shape[0]
                )
                gaussians_mem_mb = (
                    gaussians_for_level.numel() * gaussians_for_level.element_size() / 1024**2
                    if torch.is_tensor(gaussians_for_level)
                    else 0.0
                )
            else:
                gaussian_count = 0
                gaussians_mem_mb = 0.0

            level_gaussians_mem_mb.append(gaussians_mem_mb)
            print(f"scene:{scene}, level:{level}, gaussians_mem:{gaussians_mem_mb:.2f} MB")

            level_gaussian_counts.append(gaussian_count)

            if self.test_cfg.save_gaussian:
                save_path = (
                    Path(get_cfg()["output_dir"])
                    / "gaussians"
                    / (scene + f"_level{level}.ply")
                )
                save_gaussian_ply(gaussians_for_level, visualization_dump, batch, save_path)

            output = None
            if not self.train_cfg.forward_depth_only:
                with self.benchmarker.time("decoder", num_calls=v):
                    decoder_start_time = time.time() if self.test_cfg.compute_scores else None

                    camera_poses = batch["target"]["extrinsics"]

                    if self.test_cfg.stablize_camera:
                        stable_poses = render_stabilization_path(
                            camera_poses[0].detach().cpu().numpy(),
                            k_size=self.test_cfg.stab_camera_kernel,
                        )

                        stable_poses = list(
                            map(
                                lambda x: np.concatenate(
                                    (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                                ),
                                stable_poses,
                            )
                        )
                        stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                            camera_poses
                        )
                        camera_poses = stable_poses.unsqueeze(0)

                    if self.test_cfg.render_chunk_size is not None:
                        chunk_size = self.test_cfg.render_chunk_size
                        num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                        for i in range(num_chunks):
                            start = chunk_size * i
                            end = chunk_size * (i + 1)

                            render_intrinsics = batch["target"]["intrinsics"]
                            render_near = batch["target"]["near"]
                            render_far = batch["target"]["far"]
                            curr_output = self.decoder.forward(
                                gaussians_for_level,
                                camera_poses[:, start:end],
                                render_intrinsics[:, start:end],
                                render_near[:, start:end],
                                render_far[:, start:end],
                                (h, w),
                                depth_mode='depth'
                            )

                            if i == 0:
                                output = curr_output
                            else:
                                # ignore depth
                                output.color = torch.cat((output.color, curr_output.color), dim=1)

                                if output.depth is not None and curr_output.depth is not None:
                                    output.depth = torch.cat(
                                        (output.depth, curr_output.depth), dim=1
                                    )
                    else:
                        output = self.decoder.forward(
                            gaussians_for_level,
                            camera_poses,
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (h, w),
                            depth_mode='depth'
                        )

                    if self.test_cfg.compute_scores and decoder_start_time is not None:
                        decoder_total_time += time.time() - decoder_start_time
                        decoder_time_current_level = time.time() - decoder_start_time
                        decoder_time_level.append(decoder_time_current_level)  # per view time
                        # print(camera_poses.shape)

                output_per_level.append(output)

            # save depth
            # print(pred_depths)
            # print(self.train_cfg.forward_depth_only)
            if self.test_cfg.save_depth:
                if self.train_cfg.forward_depth_only:
                    if pred_depths is None:
                        continue
                    depth = pred_depths[0].cpu().detach()  # [V, H, W]
                else:
                    depth = output.depth[0].cpu().detach()  # [V, H, W]
                    # if visualization_dump is None or "depth" not in visualization_dump:
                    #     continue
                    # depth = visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()  # [V, H, W]

                # index = batch["context"]["index"][0]
                index = batch['target']['index'][0]

                if self.test_cfg.save_depth_concat_img:
                    # concat (img0, img1, depth0, depth1)
                    image = batch["context"]["image"][0]  # [V, 3, H, W] in [0,1]
                    image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                    image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                    depth_concat = []

                for idx, depth_i in zip(index, depth):
                    depth_viz = viz_depth_tensor(1.0 / depth_i, return_numpy=True)  # [H, W, 3]

                    if self.test_cfg.save_depth_concat_img:
                        depth_concat.append(depth_viz)

                    save_path = path / "images" / scene / "depth" / f"level{level}_{idx:0>6}.png"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(depth_viz).save(save_path)

                    # save depth as npy
                    if self.test_cfg.save_depth_npy:
                        depth_npy = depth_i.detach().cpu().numpy()
                        save_path = path / "images" / scene / "depth" / f"level{level}_{idx:0>6}.npy"
                        save_dir = os.path.dirname(save_path)
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(save_path, depth_npy)

                if self.test_cfg.save_depth_concat_img:
                    depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                    concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                    save_path = path / "images" / scene / "depth" / f"img_depth_{scene}_level{level}.png"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(concat).save(save_path)

            if self.train_cfg.forward_depth_only or output is None:
                continue

            images_prob = output.color[0]
            rgb_gt = batch["target"]["image"][0]

            # Save images.
            if self.test_cfg.save_image:
                if self.test_cfg.save_gt_image:
                    for index, color, gt in zip(batch["target"]["index"][0], images_prob, rgb_gt):
                        save_image(color, path / "images" / scene / f"color/{index:0>6}_level{level}.png")
                        if level == len(multiscale_gaussians) - 1:
                            save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
                else:
                    for index, color in zip(batch["target"]["index"][0], images_prob):
                        save_image(color, path / "images" / scene / f"color/{index:0>6}_level{level}.png")
            # save video

            if self.test_cfg.save_video:
                frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
                save_video(
                    [a for a in images_prob],
                    path / "videos" / f"{scene}_frame_{frame_str}_level{level}.mp4",
                )


            # compute scores
            if self.test_cfg.compute_scores:
                if batch_idx < self.test_cfg.eval_time_skip_steps:
                    self.time_skip_steps_dict["decoder"] += v

                rgb = images_prob
                psnr_value = compute_psnr(rgb_gt, rgb).mean().item()
                ssim_value = compute_ssim(rgb_gt, rgb).mean().item()
                lpips_value = compute_lpips(rgb_gt, rgb).mean().item()

                level_psnr_key = f"psnr_level{level}"
                level_ssim_key = f"ssim_level{level}"
                level_lpips_key = f"lpips_level{level}"

                if level_psnr_key not in self.test_step_outputs:
                    self.test_step_outputs[level_psnr_key] = []
                if level_ssim_key not in self.test_step_outputs:
                    self.test_step_outputs[level_ssim_key] = []
                if level_lpips_key not in self.test_step_outputs:
                    self.test_step_outputs[level_lpips_key] = []

                self.test_step_outputs[level_psnr_key].append(psnr_value)
                self.test_step_outputs[level_ssim_key].append(ssim_value)
                self.test_step_outputs[level_lpips_key].append(lpips_value)

                # keep original keys for the final level (backward compatibility)
                if level == len(multiscale_gaussians) - 1:
                    if "psnr" not in self.test_step_outputs:
                        self.test_step_outputs["psnr"] = []
                    if "ssim" not in self.test_step_outputs:
                        self.test_step_outputs["ssim"] = []
                    if "lpips" not in self.test_step_outputs:
                        self.test_step_outputs["lpips"] = []

                    decoder_time_concat = decoder_time_current_level
                    gaussian_count_concat = level_gaussian_counts[-1] if level_gaussian_counts else 0
                    self.test_step_outputs["psnr"].append(psnr_value)
                    self.test_step_outputs["ssim"].append(ssim_value)
                    self.test_step_outputs["lpips"].append(lpips_value)

                print(f"scene:{scene}, level:{level}, psnr:{psnr_value}")
                print(f"scene:{scene}, level:{level}, ssim:{ssim_value}")
                print(f"scene:{scene}, level:{level}, lpips:{lpips_value}")

                with open(path / "scene_lpips.txt", "a") as f:
                    f.write(f"{batch['scene'][0]} level{level}: {lpips_value}\n")

        # Record inference time and update statistics (per test step)
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1

            if 'decoder_times' not in self.inference_stats:
                self.inference_stats['decoder_times'] = []
            self.inference_stats['decoder_times'].append(decoder_total_time)
            self.inference_stats.setdefault('decoder_times_concat', []).append(decoder_time_concat)

            for level, level_time in enumerate(decoder_time_level):
                self.inference_stats.setdefault(f'decoder_times_level{level}', []).append(level_time)

            if level_gaussian_counts:
                self.inference_stats['gaussian_counts'].append(np.mean(level_gaussian_counts))
                for level, lc in enumerate(level_gaussian_counts):
                    self.inference_stats.setdefault(f'gaussian_counts_level{level}', []).append(lc)
                self.inference_stats.setdefault('gaussian_counts_concat', []).append(level_gaussian_counts[-1])
                if level_gaussians_mem_mb:
                    self.inference_stats.setdefault('gaussians_mem_mb', []).append(np.mean(level_gaussians_mem_mb))
                    for level, gm in enumerate(level_gaussians_mem_mb):
                        self.inference_stats.setdefault(f'gaussians_mem_mb_level{level}', []).append(gm)
                 # Track level 0 Gaussian count separately
            else:
                self.inference_stats['gaussian_counts'].append(0)
                self.inference_stats.setdefault('gaussians_mem_mb', []).append(0)

            inference_time = time.time() - inference_start_time if inference_start_time is not None else 0
            self.inference_stats['inference_times'].append(inference_time)
            self.inference_stats['step_count'] += 1

            encoder_t = self.inference_stats['encoder_times'][-1] if self.inference_stats['encoder_times'] else 0
            decoder_t = self.inference_stats['decoder_times'][-1] if self.inference_stats['decoder_times'] else 0
            gaussian_count = self.inference_stats['gaussian_counts'][-1] if self.inference_stats['gaussian_counts'] else 0

            print(
                f"[Step {self.inference_stats['step_count']}] Scene: {scene} | "
                f"Total: {inference_time:.3f}s | Encoder: {encoder_t:.3f}s | Decoder(all-levels): {decoder_t:.3f}s | "
                f"Gaussians(avg): {gaussian_count:.0f} | EncoderMem: {encoder_mem_mb:.2f}MB | Levels: {len(multiscale_gaussians)}"
            )

            if self.inference_stats['step_count'] == 100:
                avg_memory_allocated = np.mean(self.inference_stats['memory_allocated'][:100]) if self.inference_stats['memory_allocated'] else 0
                avg_memory_reserved = np.mean(self.inference_stats['memory_reserved'][:100]) if self.inference_stats['memory_reserved'] else 0
                avg_inference_time = np.mean(self.inference_stats['inference_times'][:100]) if self.inference_stats['inference_times'] else 0
                avg_encoder_time = np.mean(self.inference_stats['encoder_times'][:100]) if 'encoder_times' in self.inference_stats and self.inference_stats['encoder_times'] else 0
                avg_decoder_time = np.mean(self.inference_stats['decoder_times'][:100]) if 'decoder_times' in self.inference_stats and self.inference_stats['decoder_times'] else 0
                avg_gaussian_count = np.mean(self.inference_stats['gaussian_counts'][:100]) if self.inference_stats['gaussian_counts'] else 0

                # Calculate average metrics for first 100 steps
                avg_psnr = np.mean(self.test_step_outputs.get('psnr', [0])[:100]) if self.test_step_outputs.get('psnr') else 0
                avg_ssim = np.mean(self.test_step_outputs.get('ssim', [0])[:100]) if self.test_step_outputs.get('ssim') else 0
                avg_lpips = np.mean(self.test_step_outputs.get('lpips', [0])[:100]) if self.test_step_outputs.get('lpips') else 0

                print(f"\n{'='*70}")
                print(f"STATISTICS SUMMARY (First 100 steps)")
                print(f"{'='*70}")
                print(f"\n--- Timing Statistics ---")
                print(f"  Total Inference Time:  {avg_inference_time:.4f} seconds/step")
                print(f"  Encoder Time:          {avg_encoder_time:.4f} seconds/step ({avg_encoder_time/avg_inference_time*100:.1f}%)" if avg_inference_time > 0 else "  Encoder Time: N/A")
                print(f"  Decoder Time:          {avg_decoder_time:.4f} seconds/step ({avg_decoder_time/avg_inference_time*100:.1f}%)" if avg_inference_time > 0 else "  Decoder Time: N/A")
                print(f"  FPS (approx):          {1.0/avg_inference_time:.2f}" if avg_inference_time > 0 else "  FPS: N/A")

                print(f"\n--- Memory Statistics ---")
                print(f"  GPU Memory Allocated:  {avg_memory_allocated:.1f} MB")
                print(f"  GPU Memory Reserved:   {avg_memory_reserved:.1f} MB")

                print(f"\n--- Model Statistics ---")
                print(f"  Average Gaussian Pts:  {avg_gaussian_count:.0f}")

                print(f"\n--- Quality Metrics ---")
                print(f"  PSNR:   {avg_psnr:.4f}")
                print(f"  SSIM:   {avg_ssim:.4f}")
                print(f"  LPIPS:  {avg_lpips:.4f}")
                print(f"{'='*70}\n")



    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            # Print final statistics summary
            if hasattr(self, 'inference_stats') and self.inference_stats['step_count'] > 0:
                print("\n" + "=" * 80)
                print(" " * 25 + "FINAL TEST STATISTICS SUMMARY")
                print("=" * 80)

                # Timing Statistics
                print("\n--- TIMING STATISTICS ---")
                if self.inference_stats['inference_times']:
                    avg_inference_time = np.mean(self.inference_stats['inference_times'])
                    median_inference_time = np.median(self.inference_stats['inference_times'])
                    min_inference_time = np.min(self.inference_stats['inference_times'])
                    max_inference_time = np.max(self.inference_stats['inference_times'])
                    total_inference_time = np.sum(self.inference_stats['inference_times'])
                    std_inference_time = np.std(self.inference_stats['inference_times'])

                    print(f"  Total Inference Time:      {total_inference_time:.2f} seconds ({total_inference_time/60:.2f} minutes)")
                    print(f"  Average Inference Time:    {avg_inference_time:.4f} seconds/step")
                    print(f"  Median Inference Time:     {median_inference_time:.4f} seconds/step")
                    print(f"  Std Dev:                   {std_inference_time:.4f} seconds")
                    print(f"  Min/Max:                   {min_inference_time:.4f}s / {max_inference_time:.4f}s")
                    print(f"  Average FPS:               {1.0/avg_inference_time:.2f}")

                if 'encoder_times' in self.inference_stats and self.inference_stats['encoder_times']:
                    avg_encoder_time = np.mean(self.inference_stats['encoder_times'])
                    median_encoder_time = np.median(self.inference_stats['encoder_times'])
                    total_encoder_time = np.sum(self.inference_stats['encoder_times'])

                    print(f"\n  Encoder Time (Total):      {total_encoder_time:.2f} seconds")
                    print(f"  Encoder Time (Average):    {avg_encoder_time:.4f} seconds/step")
                    print(f"  Encoder Time (Median):     {median_encoder_time:.4f} seconds/step")

                    if self.inference_stats['inference_times']:
                        encoder_pct = (avg_encoder_time / avg_inference_time * 100) if avg_inference_time > 0 else 0
                        print(f"  Encoder Percentage:        {encoder_pct:.1f}% of total")

                if 'decoder_times' in self.inference_stats and self.inference_stats['decoder_times']:
                    avg_decoder_time = np.mean(self.inference_stats['decoder_times'])
                    median_decoder_time = np.median(self.inference_stats['decoder_times'])
                    total_decoder_time = np.sum(self.inference_stats['decoder_times'])

                    print(f"\n  Decoder Time (Total):      {total_decoder_time:.2f} seconds")
                    print(f"  Decoder Time (Average):    {avg_decoder_time:.4f} seconds/step")
                    print(f"  Decoder Time (Median):     {median_decoder_time:.4f} seconds/step")

                    if self.inference_stats['inference_times']:
                        decoder_pct = (avg_decoder_time / avg_inference_time * 100) if avg_inference_time > 0 else 0
                        print(f"  Decoder Percentage:        {decoder_pct:.1f}% of total")



                # Memory Statistics
                print("\n--- MEMORY STATISTICS ---")
                if self.inference_stats['memory_allocated']:
                    avg_memory_allocated = np.mean(self.inference_stats['memory_allocated'])
                    max_memory_allocated = np.max(self.inference_stats['memory_allocated'])
                    min_memory_allocated = np.min(self.inference_stats['memory_allocated'])
                    print(f"  GPU Memory Allocated:")
                    print(f"    Average: {avg_memory_allocated:.1f} MB")
                    print(f"    Peak:    {max_memory_allocated:.1f} MB")
                    print(f"    Min:     {min_memory_allocated:.1f} MB")

                if self.inference_stats['memory_reserved']:
                    avg_memory_reserved = np.mean(self.inference_stats['memory_reserved'])
                    max_memory_reserved = np.max(self.inference_stats['memory_reserved'])
                    min_memory_reserved = np.min(self.inference_stats['memory_reserved'])
                    print(f"\n  GPU Memory Reserved:")
                    print(f"    Average: {avg_memory_reserved:.1f} MB")
                    print(f"    Peak:    {max_memory_reserved:.1f} MB")
                    print(f"    Min:     {min_memory_reserved:.1f} MB")

                # Gaussian Statistics
                print("\n--- MODEL STATISTICS ---")
                if self.inference_stats['gaussian_counts']:
                    avg_gaussian_count = np.mean(self.inference_stats['gaussian_counts'])
                    median_gaussian_count = np.median(self.inference_stats['gaussian_counts'])
                    min_gaussian_count = np.min(self.inference_stats['gaussian_counts'])
                    max_gaussian_count = np.max(self.inference_stats['gaussian_counts'])
                    std_gaussian_count = np.std(self.inference_stats['gaussian_counts'])
                    print(f"  Gaussian Points:")
                    print(f"    Average: {avg_gaussian_count:.0f}")
                    print(f"    Median:  {median_gaussian_count:.0f}")
                    print(f"    Std Dev: {std_gaussian_count:.0f}")
                    print(f"    Min/Max: {min_gaussian_count:.0f} / {max_gaussian_count:.0f}")

                # Quality Metrics Summary
                if self.test_step_outputs:
                    print("\n--- QUALITY METRICS ---")
                    if 'psnr' in self.test_step_outputs and self.test_step_outputs['psnr']:
                        psnr_values = self.test_step_outputs['psnr']
                        print(f"  PSNR:   {np.mean(psnr_values):.4f} (± {np.std(psnr_values):.4f})")
                    if 'ssim' in self.test_step_outputs and self.test_step_outputs['ssim']:
                        ssim_values = self.test_step_outputs['ssim']
                        print(f"  SSIM:   {np.mean(ssim_values):.4f} (± {np.std(ssim_values):.4f})")
                    if 'lpips' in self.test_step_outputs and self.test_step_outputs['lpips']:
                        lpips_values = self.test_step_outputs['lpips']
                        print(f"  LPIPS:  {np.mean(lpips_values):.4f} (± {np.std(lpips_values):.4f})")

                print(f"\nTotal Test Steps: {self.inference_stats['step_count']}")
                print("=" * 80)

                # Save detailed statistics to file
                stats_summary = {
                    'total_steps': self.inference_stats['step_count'],
                    'memory_stats': {
                        'avg_allocated_mb': float(np.mean(self.inference_stats['memory_allocated'])) if self.inference_stats['memory_allocated'] else 0,
                        'peak_allocated_mb': float(np.max(self.inference_stats['memory_allocated'])) if self.inference_stats['memory_allocated'] else 0,
                        'min_allocated_mb': float(np.min(self.inference_stats['memory_allocated'])) if self.inference_stats['memory_allocated'] else 0,
                        'avg_reserved_mb': float(np.mean(self.inference_stats['memory_reserved'])) if self.inference_stats['memory_reserved'] else 0,
                        'peak_reserved_mb': float(np.max(self.inference_stats['memory_reserved'])) if self.inference_stats['memory_reserved'] else 0,
                        'min_reserved_mb': float(np.min(self.inference_stats['memory_reserved'])) if self.inference_stats['memory_reserved'] else 0,
                    },
                    'inference_time_stats': {
                        'total_time_seconds': float(np.sum(self.inference_stats['inference_times'])) if self.inference_stats['inference_times'] else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['inference_times'])) if self.inference_stats['inference_times'] else 0,
                        'median_time_per_step': float(np.median(self.inference_stats['inference_times'])) if self.inference_stats['inference_times'] else 0,
                        'std_dev': float(np.std(self.inference_stats['inference_times'])) if self.inference_stats['inference_times'] else 0,
                        'min_time': float(np.min(self.inference_stats['inference_times'])) if self.inference_stats['inference_times'] else 0,
                        'max_time': float(np.max(self.inference_stats['inference_times'])) if self.inference_stats['inference_times'] else 0,
                        'fps': float(1.0 / np.mean(self.inference_stats['inference_times'])) if (self.inference_stats['inference_times'] and np.mean(self.inference_stats['inference_times']) > 0) else 0,
                    },
                    'encoder_time_stats': {
                        'total_time_seconds': float(np.sum(self.inference_stats['encoder_times'])) if ('encoder_times' in self.inference_stats and self.inference_stats['encoder_times']) else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['encoder_times'])) if ('encoder_times' in self.inference_stats and self.inference_stats['encoder_times']) else 0,
                        'median_time_per_step': float(np.median(self.inference_stats['encoder_times'])) if ('encoder_times' in self.inference_stats and self.inference_stats['encoder_times']) else 0,
                        'percentage_of_total': float((np.mean(self.inference_stats['encoder_times']) / np.mean(self.inference_stats['inference_times']) * 100)) if ('encoder_times' in self.inference_stats and self.inference_stats['encoder_times'] and self.inference_stats['inference_times'] and np.mean(self.inference_stats['inference_times']) > 0) else 0,
                        'avg_memory_mb': float(np.mean(self.inference_stats['encoder_mem_mb'])) if self.inference_stats.get('encoder_mem_mb') else 0,
                    },
                    'decoder_time_stats': {
                        'total_time_seconds': float(np.sum(self.inference_stats['decoder_times'])) if ('decoder_times' in self.inference_stats and self.inference_stats['decoder_times']) else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['decoder_times'])) if ('decoder_times' in self.inference_stats and self.inference_stats['decoder_times']) else 0,
                        'median_time_per_step': float(np.median(self.inference_stats['decoder_times'])) if ('decoder_times' in self.inference_stats and self.inference_stats['decoder_times']) else 0,
                        'percentage_of_total': float((np.mean(self.inference_stats['decoder_times']) / np.mean(self.inference_stats['inference_times']) * 100)) if ('decoder_times' in self.inference_stats and self.inference_stats['decoder_times'] and self.inference_stats['inference_times'] and np.mean(self.inference_stats['inference_times']) > 0) else 0,
                    },
                    'gaussian_stats_level0': {
                        'avg_count': float(np.mean(self.inference_stats['gaussian_counts_level0'])) if self.inference_stats['gaussian_counts_level0'] else 0,
                        'median_count': float(np.median(self.inference_stats['gaussian_counts_level0'])) if self.inference_stats['gaussian_counts_level0'] else 0,
                        'std_dev': float(np.std(self.inference_stats['gaussian_counts_level0'])) if self.inference_stats['gaussian_counts_level0'] else 0,
                        'min_count': float(np.min(self.inference_stats['gaussian_counts_level0'])) if self.inference_stats['gaussian_counts_level0'] else 0,
                        'max_count': float(np.max(self.inference_stats['gaussian_counts_level0'])) if self.inference_stats['gaussian_counts_level0'] else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['decoder_times_level0'])) if self.inference_stats.get('decoder_times_level0') else 0,
                        'avg_memory_mb': float(np.mean(self.inference_stats['gaussians_mem_mb_level0'])) if self.inference_stats.get('gaussians_mem_mb_level0') else 0,
                    },
                    'gaussian_stats_level1': {
                        'avg_count': float(np.mean(self.inference_stats['gaussian_counts_level1'])) if self.inference_stats['gaussian_counts_level1'] else 0,
                        'median_count': float(np.median(self.inference_stats['gaussian_counts_level1'])) if self.inference_stats['gaussian_counts_level1'] else 0,
                        'std_dev': float(np.std(self.inference_stats['gaussian_counts_level1'])) if self.inference_stats['gaussian_counts_level1'] else 0,
                        'min_count': float(np.min(self.inference_stats['gaussian_counts_level1'])) if self.inference_stats['gaussian_counts_level1'] else 0,
                        'max_count': float(np.max(self.inference_stats['gaussian_counts_level1'])) if self.inference_stats['gaussian_counts_level1'] else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['decoder_times_level1'])) if self.inference_stats.get('decoder_times_level1') else 0,
                        'avg_memory_mb': float(np.mean(self.inference_stats['gaussians_mem_mb_level1'])) if self.inference_stats.get('gaussians_mem_mb_level1') else 0,
                    },
                    'gaussian_stats_level2': {
                        'avg_count': float(np.mean(self.inference_stats['gaussian_counts_level2'])) if self.inference_stats['gaussian_counts_level2'] else 0,
                        'median_count': float(np.median(self.inference_stats['gaussian_counts_level2'])) if self.inference_stats['gaussian_counts_level2'] else 0,
                        'std_dev': float(np.std(self.inference_stats['gaussian_counts_level2'])) if self.inference_stats['gaussian_counts_level2'] else 0,
                        'min_count': float(np.min(self.inference_stats['gaussian_counts_level2'])) if self.inference_stats['gaussian_counts_level2'] else 0,
                        'max_count': float(np.max(self.inference_stats['gaussian_counts_level2'])) if self.inference_stats['gaussian_counts_level2'] else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['decoder_times_level2'])) if self.inference_stats.get('decoder_times_level2') else 0,
                        'avg_memory_mb': float(np.mean(self.inference_stats['gaussians_mem_mb_level2'])) if self.inference_stats.get('gaussians_mem_mb_level2') else 0,
                    },
                    'gaussian_stats_level3': {
                        'avg_count': float(np.mean(self.inference_stats['gaussian_counts_level3'])) if self.inference_stats['gaussian_counts_level3'] else 0,
                        'median_count': float(np.median(self.inference_stats['gaussian_counts_level3'])) if self.inference_stats['gaussian_counts_level3'] else 0,
                        'std_dev': float(np.std(self.inference_stats['gaussian_counts_level3'])) if self.inference_stats['gaussian_counts_level3'] else 0,
                        'min_count': float(np.min(self.inference_stats['gaussian_counts_level3'])) if self.inference_stats['gaussian_counts_level3'] else 0,
                        'max_count': float(np.max(self.inference_stats['gaussian_counts_level3'])) if self.inference_stats['gaussian_counts_level3'] else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['decoder_times_level3'])) if self.inference_stats.get('decoder_times_level3') else 0,
                        'avg_memory_mb': float(np.mean(self.inference_stats['gaussians_mem_mb_level3'])) if self.inference_stats.get('gaussians_mem_mb_level3') else 0,
                    },
                    'gaussian_stats_level4': {
                        'avg_count': float(np.mean(self.inference_stats['gaussian_counts_level4'])) if self.inference_stats['gaussian_counts_level4'] else 0,
                        'median_count': float(np.median(self.inference_stats['gaussian_counts_level4'])) if self.inference_stats['gaussian_counts_level4'] else 0,
                        'std_dev': float(np.std(self.inference_stats['gaussian_counts_level4'])) if self.inference_stats['gaussian_counts_level4'] else 0,
                        'min_count': float(np.min(self.inference_stats['gaussian_counts_level4'])) if self.inference_stats['gaussian_counts_level4'] else 0,
                        'max_count': float(np.max(self.inference_stats['gaussian_counts_level4'])) if self.inference_stats['gaussian_counts_level4'] else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['decoder_times_level4'])) if self.inference_stats.get('decoder_times_level4') else 0,
                        'avg_memory_mb': float(np.mean(self.inference_stats['gaussians_mem_mb_level4'])) if self.inference_stats.get('gaussians_mem_mb_level4') else 0,
                    },
                    "gaussian_stats_concat": {
                        'avg_count': float(np.mean(self.inference_stats['gaussian_counts_concat'])) if self.inference_stats['gaussian_counts_concat'] else 0,
                        'median_count': float(np.median(self.inference_stats['gaussian_counts_concat'])) if self.inference_stats['gaussian_counts_concat'] else 0,
                        'std_dev': float(np.std(self.inference_stats['gaussian_counts_concat'])) if self.inference_stats['gaussian_counts_concat'] else 0,
                        'min_count': float(np.min(self.inference_stats['gaussian_counts_concat'])) if self.inference_stats['gaussian_counts_concat'] else 0,
                        'max_count': float(np.max(self.inference_stats['gaussian_counts_concat'])) if self.inference_stats['gaussian_counts_concat'] else 0,
                        'avg_time_per_step': float(np.mean(self.inference_stats['decoder_times_concat'])) if self.inference_stats.get('decoder_times_concat') else 0,
                    },


                    'quality_metrics': {
                        'psnr': {
                            'mean': float(np.mean(self.test_step_outputs.get('psnr', [0]))) if self.test_step_outputs.get('psnr') else 0,
                            'std': float(np.std(self.test_step_outputs.get('psnr', [0]))) if self.test_step_outputs.get('psnr') else 0,
                        },
                        'ssim': {
                            'mean': float(np.mean(self.test_step_outputs.get('ssim', [0]))) if self.test_step_outputs.get('ssim') else 0,
                            'std': float(np.std(self.test_step_outputs.get('ssim', [0]))) if self.test_step_outputs.get('ssim') else 0,
                        },
                        'lpips': {
                            'mean': float(np.mean(self.test_step_outputs.get('lpips', [0]))) if self.test_step_outputs.get('lpips') else 0,
                            'std': float(np.std(self.test_step_outputs.get('lpips', [0]))) if self.test_step_outputs.get('lpips') else 0,
                        }
                    }
                }

                with (out_dir / "inference_stats_summary.json").open("w") as f:
                    json.dump(stats_summary, f, indent=2)

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax, gaussian_softmax_list, mask_list, _ = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
            max_levels=self.max_levels
        )


        multiscale_gaussians = []
        for i, m in enumerate(mask_list):

            single_scale_gaussian = Gaussians(
            means=gaussians_softmax["gaussians"].means[m].expand(gaussians_softmax["gaussians"].means.shape[0], -1, -1) if len(gaussians_softmax["gaussians"].means.shape) > 1 else gaussians_softmax["gaussians"].means[m].expand(gaussians_softmax["gaussians"].means.shape[0], -1),
            covariances=gaussians_softmax["gaussians"].covariances[m].expand(gaussians_softmax["gaussians"].covariances.shape[0], -1, -1, -1) if len(gaussians_softmax["gaussians"].covariances.shape) > 1 else gaussians_softmax["gaussians"].covariances[m].expand(gaussians_softmax["gaussians"].covariances.shape[0], -1, -1),
            harmonics=gaussians_softmax["gaussians"].harmonics[m].expand(gaussians_softmax["gaussians"].harmonics.shape[0], -1, -1, -1) if len(gaussians_softmax["gaussians"].harmonics.shape) > 1 else gaussians_softmax["gaussians"].harmonics[m].expand(gaussians_softmax["gaussians"].harmonics.shape[0], -1, -1),
            opacities=gaussians_softmax["gaussians"].opacities[m].expand(gaussians_softmax["gaussians"].opacities.shape[0], -1) if len(gaussians_softmax["gaussians"].opacities.shape) > 1 else gaussians_softmax["gaussians"].opacities[m].expand(gaussians_softmax["gaussians"].opacities.shape[0], -1),
            levels=gaussians_softmax["gaussians"].levels[m].expand(gaussians_softmax["gaussians"].levels.shape[0], -1) if len(gaussians_softmax["gaussians"].levels.shape) > 1 else gaussians_softmax["gaussians"].levels[m].expand(gaussians_softmax["gaussians"].levels.shape[0], -1)
            )
            multiscale_gaussians.append({"gaussians": single_scale_gaussian, "depths": gaussians_softmax["depths"]})

        if len(multiscale_gaussians) == 0:
            multiscale_gaussians = [gaussians_softmax]

        for level, level_gaussians in enumerate(multiscale_gaussians):
            pred_depths = None
            gaussians_level = level_gaussians

            if isinstance(gaussians_level, dict):
                pred_depths = gaussians_level.get("depths", None)
                if "depth" in batch["context"]:
                    depth_gt = batch["context"]["depth"]  # [B, V, H, W]
                gaussians_level = gaussians_level["gaussians"]

            if not self.train_cfg.forward_depth_only:
                output_softmax = self.decoder.forward(
                    gaussians_level,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )
                rgb_softmax = output_softmax.color[0]

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                psnr = compute_psnr(rgb_gt, rgb_softmax).mean()
                lpips = compute_lpips(rgb_gt, rgb_softmax).mean()
                ssim = compute_ssim(rgb_gt, rgb_softmax).mean()

                self.log(f"val/psnr_val_level{level}", psnr)
                self.log(f"val/lpips_val_level{level}", lpips)
                self.log(f"val/ssim_val_level{level}", ssim)

                # keep original tags for the last level (backward compatibility)
                if level == len(multiscale_gaussians) - 1:
                    self.log("val/psnr_val", psnr)
                    self.log("val/lpips_val", lpips)
                    self.log("val/ssim_val", ssim)

            # viz depth
            if pred_depths is not None:
                # only visualize predicted depth
                pred_depths = pred_depths[0]  # [V, H, W]

                # gaussian downsample
                if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                    pred_depths = F.interpolate(
                        pred_depths.unsqueeze(1),
                        size=batch["context"]["image"].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(1)

                inverse_depth_pred = 1.0 / pred_depths

                concat = []
                for i in range(inverse_depth_pred.size(0)):
                    concat.append(inverse_depth_pred[i])

                concat = torch.cat(concat, dim=1)  # [H, W*N]

                depth_viz = viz_depth_tensor(concat.cpu().detach())  # [3, H, W*N]

                # also concat images
                input_images = batch["context"]["image"][0]  # [N, 3, H, W]
                concat_img = [img for img in input_images]
                concat_img = torch.cat(concat_img, dim=-1) * 255  # [3, H, W*N]

                concat = torch.cat(
                    (concat_img.cpu().detach(), depth_viz), dim=1
                )  # [3, H*2, W*N]

                self.logger.log_image(
                    f"depth_level{level}",
                    [concat],
                    step=self.global_step,
                    caption=batch["scene"],
                )

                # keep original tag for the last level (backward compatibility)
                if level == len(multiscale_gaussians) - 1:
                    self.logger.log_image(
                        "depth",
                        [concat],
                        step=self.global_step,
                        caption=batch["scene"],
                    )

        # if not self.train_cfg.forward_depth_only:
        #     # Construct comparison image.
        #     comparison = hcat(
        #         add_label(vcat(*batch["context"]["image"][0]), "Context"),
        #         add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
        #         add_label(vcat(*rgb_softmax), "Target (Prediction)"),
        #         # add_label(vcat(batch["context"]["image"][0]), "Context"),
        #         # add_label(vcat(rgb_gt), "Target (Ground Truth)"),
        #         # add_label(vcat(rgb_softmax), "Target (Prediction)"),
        #     )
        #     self.logger.log_image(
        #         "comparison",
        #         [prep_image(add_border(comparison))],
        #         step=self.global_step,
        #         caption=batch["scene"],
        #     )

        #     if not self.train_cfg.no_log_projections:
        #         # Render projections and construct projection image.
        #         projections = hcat(
        #             *render_projections(
        #                 gaussians_softmax,
        #                 256,
        #                 extra_label="(Prediction)",
        #             )[0]
        #         )
        #         self.logger.log_image(
        #             "projection",
        #             [prep_image(add_border(projections))],
        #             step=self.global_step,
        #         )

        #         # Draw cameras.
        #         cameras = hcat(*render_cameras(batch, 256))
        #         self.logger.log_image(
        #             "cameras", [prep_image(add_border(cameras))], step=self.global_step
        #         )

        #     if self.encoder_visualizer is not None:
        #         for k, image in self.encoder_visualizer.visualize(
        #             batch["context"], self.global_step
        #         ).items():
        #             self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        #     # Run video validation step.
        #     if not self.train_cfg.no_viz_video:
        #         self.render_video_interpolation(batch)
        #         self.render_video_wobble(batch)
        #         if self.train_cfg.extended_visualization:
        #             self.render_video_interpolation_exaggerated(batch)

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self.encoder)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        pred_depths = None
        depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            # dataset_cfg=self.eval_data_cfg
        )
        scores_dict = {}

        if not self.train_cfg.forward_depth_only:
            for score_tag in ("psnr", "ssim", "lpips"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        # evaluate depth
        if self.train_cfg.viz_depth:
            for score_tag in ("abs_rel", "rmse", "a1"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
            enumerate(full_testsets),
            total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break

            batch = self.data_shim(batch)
            batch = self.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1

            with self.benchmarker.time("encoder"):
                gaussians_probabilistic, gaussians_probabilistic_list, mask_list,  _ = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                    max_levels=self.max_levels
                )

                # if len(encoder_out_prob) == 3:
                #     (
                #         gaussians_probabilistic,
                #         gaussians_probabilistic_list,
                #         mask_list,
                #     ) = encoder_out_prob
                # else:
                #     gaussians_probabilistic, gaussians_probabilistic_list = encoder_out_prob
                #     mask_list = None

                if mask_list is not None:
                    multiscale_gaussians = []
                    for i, m in enumerate(mask_list):

                        single_scale_gaussian = Gaussians(
                        means=gaussians_probabilistic["gaussians"].means[m].expand(gaussians_probabilistic["gaussians"].means.shape[0], -1, -1) if len(gaussians_probabilistic["gaussians"].means.shape) > 1 else gaussians_probabilistic["gaussians"].means[m].expand(gaussians_probabilistic["gaussians"].means.shape[0], -1),
                        covariances=gaussians_probabilistic["gaussians"].covariances[m].expand(gaussians_probabilistic["gaussians"].covariances.shape[0], -1, -1, -1) if len(gaussians_probabilistic["gaussians"].covariances.shape) > 1 else gaussians_probabilistic["gaussians"].covariances[m].expand(gaussians_probabilistic["gaussians"].covariances.shape[0], -1, -1),
                        harmonics=gaussians_probabilistic["gaussians"].harmonics[m].expand(gaussians_probabilistic["gaussians"].harmonics.shape[0], -1, -1, -1) if len(gaussians_probabilistic["gaussians"].harmonics.shape) > 1 else gaussians_probabilistic["gaussians"].harmonics[m].expand(gaussians_probabilistic["gaussians"].harmonics.shape[0], -1, -1),
                        opacities=gaussians_probabilistic["gaussians"].opacities[m].expand(gaussians_probabilistic["gaussians"].opacities.shape[0], -1) if len(gaussians_probabilistic["gaussians"].opacities.shape) > 1 else gaussians_probabilistic["gaussians"].opacities[m].expand(gaussians_probabilistic["gaussians"].opacities.shape[0], -1),
                        levels=gaussians_probabilistic["gaussians"].levels[m].expand(gaussians_probabilistic["gaussians"].levels.shape[0], -1) if len(gaussians_probabilistic["gaussians"].levels.shape) > 1 else gaussians_probabilistic["gaussians"].levels[m].expand(gaussians_probabilistic["gaussians"].levels.shape[0], -1)
                        )
                        multiscale_gaussians.append({"gaussians": single_scale_gaussian, "depths": gaussians_probabilistic["depths"]})
                else:
                    multiscale_gaussians = [gaussians_probabilistic]
                multiscale_gaussians_prob = multiscale_gaussians
                if len(multiscale_gaussians_prob) == 0:
                    multiscale_gaussians_prob = [gaussians_probabilistic]

            if not self.train_cfg.forward_depth_only:
                rgbs = []
                tags = []

                for level, level_gaussians in enumerate(multiscale_gaussians_prob):
                    gaussians_prob_level = level_gaussians
                    if isinstance(gaussians_prob_level, dict):
                        pred_depths = gaussians_prob_level.get("depths", pred_depths)
                        if "depth" in batch["context"]:
                            depth_gt = batch["context"]["depth"]
                        gaussians_prob_level = gaussians_prob_level["gaussians"]

                    with self.benchmarker.time("decoder", num_calls=v):
                        output_probabilistic = self.decoder.forward(
                            gaussians_prob_level,
                            batch["target"]["extrinsics"],
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (h, w),
                        )

                    if batch_idx < time_skip_first_n_steps:
                        time_skip_steps_dict["decoder"] += v

                    rgbs.append(output_probabilistic.color[0])
                    tags.append(f"probabilistic_level{level}")

                    # backward compatibility: keep final level under "probabilistic"
                    if level == len(multiscale_gaussians_prob) - 1:
                        rgbs.append(output_probabilistic.color[0])
                        tags.append("probabilistic")

                if self.train_cfg.eval_deterministic:
                    gaussians_deterministic, gaussians_deterministic_list, mask_list, _ = self.encoder(
                        batch["context"],
                        self.global_step,
                        deterministic=True,
                        max_levels=self.max_levels
                    )

                    # if len(encoder_out_det) == 3:
                    #     (
                    #         gaussians_deterministic,
                    #         gaussians_deterministic_list,
                    #         _,
                    #     ) = encoder_out_det
                    # else:
                    #     gaussians_deterministic, gaussians_deterministic_list = encoder_out_det

                    if isinstance(gaussians_deterministic, dict):
                        gaussians_deterministic = gaussians_deterministic["gaussians"]

                    output_deterministic = self.decoder.forward(
                        gaussians_deterministic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                    rgbs.append(output_deterministic.color[0])
                    tags.append("deterministic")

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                for tag, rgb in zip(tags, rgbs):
                    if tag not in scores_dict["psnr"]:
                        scores_dict["psnr"][tag] = []
                    if tag not in scores_dict["lpips"]:
                        scores_dict["lpips"][tag] = []
                    if tag not in scores_dict["ssim"]:
                        scores_dict["ssim"][tag] = []

                    scores_dict["psnr"][tag].append(
                        compute_psnr(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["lpips"][tag].append(
                        compute_lpips(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["ssim"][tag].append(
                        compute_ssim(rgb_gt, rgb).mean().item()
                    )

        # summarise scores and log to logger
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]) :]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        print(f"Eval total time cost: {overall_eval_time:.3f}s")
        self.log("test/runtime_all", overall_eval_time)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob, gaussians_prob_list, _ , _= self.encoder(batch["context"], self.global_step, False, max_levels=self.max_levels)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        # visualizations = {
        #     f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        # }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        # try:
        #     wandb.log(visualizations)
        # except Exception:
        #     assert isinstance(self.logger, LocalLogger)
        #     for key, value in visualizations.items():
        #         tensor = value._prepare_video(value.data)
        #         clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
        #         dir = LOG_PATH / key
        #         dir.mkdir(exist_ok=True, parents=True)
        #         clip.write_videofile(
        #             str(dir / f"{self.global_step:0>6}.mp4"), logger=None
        #         )

    def configure_optimizers(self):
        parameter_groups = {
            "pretrained": [],
            "level_embedding": [],
            "default": [],
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "level_embedding" in name:
                parameter_groups["level_embedding"].append(param)
            elif "pretrained" in name and self.optimizer_cfg.lr_monodepth > 0:
                parameter_groups["pretrained"].append(param)
            else:
                parameter_groups["default"].append(param)

        groups = []
        max_lrs = []
        for group_name, lr in (
            ("pretrained", self.optimizer_cfg.lr_monodepth),
            ("level_embedding", self.optimizer_cfg.lr_level_embedding),
            ("default", self.optimizer_cfg.lr),
        ):
            params = parameter_groups[group_name]
            if params:
                groups.append({"params": params, "lr": lr})
                max_lrs.append(lr)

        optimizer = optim.AdamW(
            groups,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lrs,
            self.trainer.max_steps + 10,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
