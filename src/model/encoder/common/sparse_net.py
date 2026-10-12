import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
class SparseGaussianHead(nn.Module):
    def __init__(self, in_channels=164, out_channels=38):

        super().__init__()

        self.num_gaussian_parameters = out_channels

        self.conv1 = ME.MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dimension=3
        )
        self.act = ME.MinkowskiGELU()
        self.conv2 = ME.MinkowskiConvolution(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dimension=3
        )

        self.init_weights()

    def forward(self, sparse_input: ME.SparseTensor):

        x = self.conv1(sparse_input)
        x = self.act(x)
        x = self.conv2(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                try:
                    ME.utils.kaiming_normal_(m.kernel,
                                                mode='fan_out',
                                                nonlinearity='relu')
                except Exception:
                    nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

                if hasattr(m, 'bias') and m.bias is not None:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except Exception:
                        pass

            elif isinstance(m, (ME.MinkowskiInstanceNorm, ME.MinkowskiBatchNorm)):
                # Check for direct weight/bias attributes first (common in ME)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, ME.MinkowskiInstanceNorm):
            #     if hasattr(m, 'bn'):
            #         try:
            #             nn.init.constant_(m.bn.weight, 1)
            #             nn.init.constant_(m.bn.bias, 0)
            #         except Exception:
            #             pass


class SparseGaussianLevelsHead(nn.Module):
    def __init__(self, in_channels=164, out_channels=38):

        super().__init__()

        self.num_gaussian_parameters = out_channels

        self.conv1 = ME.MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dimension=3
        )
        self.act = ME.MinkowskiGELU()
        self.conv2 = ME.MinkowskiConvolution(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dimension=3
        )

        self.init_weights()

    def forward(self, sparse_input: list[ME.SparseTensor]):
        output = []
        expected_ch = self.conv1.kernel.size(1)
        for si in sparse_input:
            # sanity check: ensure feature dimension matches convolution input
            if si.F.size(1) != expected_ch:
                raise RuntimeError(
                    f"SparseGaussianLevelsHead received tensor with {si.F.size(1)} "
                    f"channels but conv expects {expected_ch} channels"
                )
            x = self.conv1(si)
            x = self.act(x)
            x = self.conv2(x)
            output.append(x)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                try:
                    ME.utils.kaiming_normal_(m.kernel,
                                                mode='fan_out',
                                                nonlinearity='relu')
                except Exception:
                    nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

                if hasattr(m, 'bias') and m.bias is not None:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except Exception:
                        pass


            elif isinstance(m, (ME.MinkowskiInstanceNorm, ME.MinkowskiBatchNorm)):
                # Check for direct weight/bias attributes first (common in ME)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, ME.MinkowskiInstanceNorm):
            #     if hasattr(m, 'bn'):
            #         try:
            #             nn.init.constant_(m.bn.weight, 1)
            #             nn.init.constant_(m.bn.bias, 0)
            #         except Exception:
            #             pass



class AttentionBlock(nn.Module):
    """"""
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}")
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        self.norm = ME.MinkowskiInstanceNorm(channels)

        self.qkv = ME.MinkowskiLinear(channels, channels * 3)
        self.proj_out = ME.MinkowskiLinear(channels, channels)

    def _attention(self, qkv: torch.Tensor):
        length, width = qkv.shape
        ch = width // (3 * self.num_heads)
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3)  # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1)  # (1, num_heads, length, ch)

        if hasattr(F, 'scaled_dot_product_attention'):
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                values = F.scaled_dot_product_attention(q, k, v)[0]
        else:
            values = F.scaled_dot_product_attention(q, k, v)[0]

        # values has shape (length, num_heads, ch)
        # after attention the tensor is (1, num_heads, length, ch)
        # reshape back to (length, num_heads * ch) via correct permutation
        values = values.permute(2, 0, 1, 3).reshape(length, -1)
        return values

    def forward(self, x: ME.SparseTensor):
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)

        feature_dense = self._attention(qkv.F)
        feature = ME.SparseTensor(
            features=feature_dense,
            coordinate_map_key=qkv.coordinate_map_key,
            coordinate_manager=qkv.coordinate_manager
        )

        output = self.proj_out(feature)
        return output + x

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dimension=3
        )
        self.norm = ME.MinkowskiInstanceNorm(out_channels)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor):
        return self.act(self.norm(self.conv(x)))

class SparseUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.upconv = ME.MinkowskiConvolutionTranspose(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dimension=3
        )
        self.norm = ME.MinkowskiInstanceNorm(out_channels)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor):
        return self.act(self.norm(self.upconv(x)))

class SparseUNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4, use_attention=False):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.use_attention = use_attention

        self.encoder_channels = []

        current_ch = in_channels
        for i in range(num_blocks):
            out_ch = 64 * (2 ** i)
            self.encoders.append(SparseConvBlock(current_ch, out_ch, kernel_size=3, stride=2))
            self.encoder_channels.append(out_ch)
            current_ch = out_ch


        bottleneck_in = current_ch
        bottleneck_out = bottleneck_in * 2

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(SparseConvBlock(bottleneck_in, bottleneck_out, kernel_size=3, stride=2))

        if use_attention:
            self.bottleneck.append(AttentionBlock(bottleneck_out))

        self.bottleneck.append(SparseConvBlock(bottleneck_out, bottleneck_out, kernel_size=3, stride=1))

        current_decoder_ch = bottleneck_out

        for i in range(num_blocks):
            decoder_out = self.encoder_channels[-1-i]

            upconv = SparseUpConvBlock(current_decoder_ch, decoder_out, kernel_size=3, stride=2)

            after_cat = ME.MinkowskiConvolution(
                decoder_out + self.encoder_channels[-1-i],
                decoder_out,
                kernel_size=1,
                stride=1,
                dimension=3
            )

            self.decoder_blocks.append(nn.ModuleList([upconv, after_cat]))

            current_decoder_ch = decoder_out

        self.final_upsample = SparseUpConvBlock(
            self.encoder_channels[0],
            out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x: ME.SparseTensor, return_intermediates: bool = False):
        """Standard UNet forward pass.

        Args:
            x: input sparse tensor
            return_intermediates: if True return a tuple ``(output, encoder_outputs)``
                where ``encoder_outputs`` is a list of tensors produced by each
                encoder block (useful for injecting attention or other processing).
        """
        encoder_outputs = []

        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)

        for layer in self.bottleneck:
            x = layer(x)

        for i, decoder_block in enumerate(self.decoder_blocks):
            upconv, after_cat = decoder_block

            x = upconv(x)

            enc_index = len(encoder_outputs) - i - 1
            if enc_index >= 0 and enc_index < len(encoder_outputs):
                x = ME.cat(x, encoder_outputs[enc_index])
                x = after_cat(x)

        output = self.final_upsample(x)

        return output

try:
    from xformers.ops import memory_efficient_attention
except Exception:
    memory_efficient_attention = None

class FlashWindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.gamma = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.001)
        # self.log_gamma = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x, coords, mask):
        """
        x: (Num_Windows, N_max, C)
        coords: (Num_Windows, N_max, 3)
        mask: (Num_Windows, N_max) bool
        """
        W, N, C = x.shape
        H = self.num_heads
        target_dtype = torch.float16 if x.is_cuda else torch.float32


        qkv = self.qkv(x).reshape(W, N, 3, H, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = [item.to(target_dtype).contiguous() for item in qkv]


        N_aligned = (N + 7) // 8 * 8
        bias = torch.zeros((W, H, N_aligned, N_aligned), dtype=target_dtype, device=x.device)


        # with torch.cuda.amp.autocast(enabled=False):

        #     dist_sq = torch.cdist(coords, coords, p=2).pow(2).unsqueeze(1)
        #     actual_bias = -self.gamma * dist_sq
        # gamma = torch.exp(self.log_gamma)
        # actual_bias = -gamma * dist_sq


        # bias[:, :, :N, :N] = actual_bias.to(target_dtype)


        if mask is not None:
            # (W, N) -> (W, 1, 1, N)
            m_bias = torch.zeros((W, 1, 1, N), device=x.device, dtype=target_dtype)
            m_bias.masked_fill_(~mask.view(W, 1, 1, N), float("-inf"))
            bias[:, :, :N, :N] += m_bias


        attn_bias = bias[:, :, :N, :N]

        if memory_efficient_attention is not None and x.is_cuda:
            try:
                out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            except (NotImplementedError, RuntimeError):
                # Fallback for unsupported xformers kernels / bias types.
                q_ = q.permute(0, 2, 1, 3).contiguous()  # [W, H, N, D]
                k_ = k.permute(0, 2, 1, 3).contiguous()  # [W, H, N, D]
                v_ = v.permute(0, 2, 1, 3).contiguous()  # [W, H, N, D]
                out = F.scaled_dot_product_attention(
                    q_, k_, v_, attn_mask=attn_bias, dropout_p=0.0
                )
                out = out.permute(0, 2, 1, 3).contiguous()  # [W, N, H, D]
        else:
            q_ = q.permute(0, 2, 1, 3).contiguous()  # [W, H, N, D]
            k_ = k.permute(0, 2, 1, 3).contiguous()  # [W, H, N, D]
            v_ = v.permute(0, 2, 1, 3).contiguous()  # [W, H, N, D]
            out = F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=attn_bias, dropout_p=0.0
            )
            out = out.permute(0, 2, 1, 3).contiguous()  # [W, N, H, D]


        out = out.to(x.dtype).reshape(W, N, C)
        return self.proj(out)


class SparseHiUNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4, use_attention=False,
                 max_levels=4, voxel_resolution=0.001, embed_level=False):
        super().__init__()
        self.unet = SparseUNetWithAttention(in_channels, out_channels, num_blocks, use_attention)

        self.max_levels = max_levels
        attention_dim = 512
        self.attn_layer = FlashWindowAttention(dim=attention_dim, num_heads=8)
        self.level_embedding = (
            nn.Embedding(max_levels + 1, attention_dim)
            if embed_level else None
        )
        if self.level_embedding is not None:
            nn.init.zeros_(self.level_embedding.weight)

        self.window_size = voxel_resolution * 16 * 16
        self.voxel_resolution = voxel_resolution

    def chunked_attn(self, win_in_feat, win_in_coord, win_mask, chunk_size=1024*8):
        num_windows = win_in_feat.shape[0]
        outputs = []

        for i in range(0, num_windows, chunk_size):
            # Slice the window batch
            feat_chunk = win_in_feat[i : i + chunk_size]
            coord_chunk = win_in_coord[i : i + chunk_size]
            mask_chunk = win_mask[i : i + chunk_size] if win_mask is not None else None

            # Process chunk
            out_chunk = self.attn_layer(feat_chunk, coord_chunk, mask_chunk)
            outputs.append(out_chunk)

        # Concatenate back to [67428, ...]
        return torch.cat(outputs, dim=0)

    def forward(self, x_list: list[ME.SparseTensor]):

        # mega_sparse_x = self.run_encoder(x_list)

        valid_list = [(i, x) for i, x in enumerate(x_list) if x is not None and len(x) > 0]
        if not valid_list: return [None] * len(x_list)


        all_feats = []
        all_coords = []
        batch_offsets = []
        current_offset = 0

        len_levels = []
        for _, x in valid_list:
            coords = x.C.clone()

            max_b = coords[:, 0].max().item()


            coords[:, 0] += current_offset
            all_coords.append(coords)
            all_feats.append(x.F)

            len_levels.append(x.F.shape[0])

            batch_offsets.append(current_offset)
            current_offset += (max_b + 1)



        mega_sparse_x = ME.SparseTensor(
            features=torch.cat(all_feats, dim=0),
            coordinates=torch.cat(all_coords, dim=0),
            tensor_stride=x_list[valid_list[0][0]].tensor_stride,
            device=x_list[valid_list[0][0]].device
        )



        encoder_outputs = []

        for encoder in self.unet.encoders:
            mega_sparse_x = encoder(mega_sparse_x)
            encoder_outputs.append(mega_sparse_x)

        for layer in self.unet.bottleneck:
            mega_sparse_x = layer(mega_sparse_x)


        device = mega_sparse_x.device
        features = mega_sparse_x.F
        voxel_resolution = self.voxel_resolution
        reverse_batch_indices = torch.empty(
            mega_sparse_x.C.shape[0], device=device, dtype=mega_sparse_x.C.dtype
        )
        voxel_resolutions = torch.empty(
            mega_sparse_x.C.shape[0], 1, device=device, dtype=features.dtype
        )
        level_indices = torch.empty(
            mega_sparse_x.C.shape[0], device=device, dtype=torch.long
        )
        for idx_in_valid, (original_idx, _) in enumerate(valid_list):
            offset = batch_offsets[idx_in_valid]
            next_offset = batch_offsets[idx_in_valid + 1] if idx_in_valid + 1 < len(batch_offsets) else current_offset

            start_b = offset
            end_b = next_offset
            mask = (mega_sparse_x.C[:, 0] >= start_b) & (mega_sparse_x.C[:, 0] < end_b)

            voxel_resolutions[mask] = voxel_resolution * (
                2 ** (self.max_levels - original_idx)
            )
            reverse_batch_indices[mask] = mega_sparse_x.C[mask, 0] - offset
            level_indices[mask] = original_idx

        # Inject scale identity after sparse convolution and immediately before
        # cross-level window attention.
        if self.level_embedding is not None:
            features = features + self.level_embedding(level_indices).to(features.dtype)

        phys_coords = mega_sparse_x.tensor_stride[0] * mega_sparse_x.C[...,1:].float() * voxel_resolutions + (mega_sparse_x.tensor_stride[0] * voxel_resolutions / 2.0)


        # key = batch_idx * Large_Number + quantized_coords

        win_indices = (phys_coords / self.window_size).floor().long()



        batch_win_coords = torch.cat([reverse_batch_indices.long()[..., None], win_indices], dim=1)
        # print(batch_win_coords)


        unique_keys, inverse_indices = torch.unique(batch_win_coords, dim=0, return_inverse=True)


        num_windows = unique_keys.shape[0]


        sorted_indices = torch.argsort(inverse_indices)
        inv_indices_sorted = inverse_indices[sorted_indices]


        counts = torch.bincount(inv_indices_sorted)

        max_pts = counts.max().item()



        cum_counts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])
        point_range = torch.arange(max_pts, device=device)
        local_idx_all = torch.cat([point_range[:c] for c in counts])


        win_in_feat = torch.zeros((num_windows, max_pts, features.shape[-1]), device=device)
        win_in_coord = torch.zeros((num_windows, max_pts, 3), device=device)
        win_mask = torch.zeros((num_windows, max_pts), device=device, dtype=torch.bool)


        win_in_feat[inv_indices_sorted, local_idx_all] = features[sorted_indices]
        win_in_coord[inv_indices_sorted, local_idx_all] = phys_coords[sorted_indices]
        win_mask[inv_indices_sorted, local_idx_all] = True



        refined_win_feat = self.chunked_attn(win_in_feat, win_in_coord, win_mask)


        refined_features = torch.zeros_like(features)
        refined_features[sorted_indices] = refined_win_feat[inv_indices_sorted, local_idx_all]


        mega_sparse_x = ME.SparseTensor(
            features=refined_features,
            coordinate_map_key=mega_sparse_x.coordinate_map_key,
            coordinate_manager=mega_sparse_x.coordinate_manager
        )



        for i, decoder_block in enumerate(self.unet.decoder_blocks):
            upconv, after_cat = decoder_block

            mega_sparse_x = upconv(mega_sparse_x)

            enc_index = len(encoder_outputs) - i - 1
            if enc_index >= 0 and enc_index < len(encoder_outputs):
                mega_sparse_x = ME.cat(mega_sparse_x, encoder_outputs[enc_index])
                mega_sparse_x = after_cat(mega_sparse_x)

        mega_output = self.unet.final_upsample(mega_sparse_x)




        final_outputs = [None] * len(x_list)
        for idx_in_valid, (original_idx, _) in enumerate(valid_list):
            offset = batch_offsets[idx_in_valid]

            next_offset = batch_offsets[idx_in_valid + 1] if idx_in_valid + 1 < len(batch_offsets) else current_offset




            final_outputs[original_idx] = self._extract_by_batch_range(mega_output, offset, next_offset)

        return final_outputs

    def _extract_by_batch_range(self, sparse_tensor, start_b, end_b):

        mask = (sparse_tensor.C[:, 0] >= start_b) & (sparse_tensor.C[:, 0] < end_b)
        new_coords = sparse_tensor.C[mask]
        new_coords[:, 0] -= start_b
        return ME.SparseTensor(
            features=sparse_tensor.F[mask],
            coordinates=new_coords,
            tensor_stride=sparse_tensor.tensor_stride,
            coordinate_manager=sparse_tensor.coordinate_manager,
            device=sparse_tensor.device
        )

#  class SparseHiUNetWithAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, num_blocks=4, use_attention=False, max_levels=4):
#         super().__init__()
#         self.unet = SparseUNetWithAttention(in_channels, out_channels, num_blocks, use_attention)

#         self.max_levels = max_levels
#         self.level_embedding = torch.nn.Embedding(max_levels+1, in_channels)  # learnable level embedding




#     def forward(self, x_list: list[ME.SparseTensor],
#                 attention_on_intermediates: bool = False):

#         valid_list = [(i, x) for i, x in enumerate(x_list) if x is not None and len(x) > 0]
#         if not valid_list: return [None] * len(x_list)


#         all_feats = []
#         all_coords = []
#         batch_offsets = []
#         current_offset = 0

#         len_levels = []
#         for _, x in valid_list:
#             coords = x.C.clone()

#             max_b = coords[:, 0].max().item()


#             coords[:, 0] += current_offset
#             all_coords.append(coords)
#             all_feats.append(x.F)

#             len_levels.append(x.F.shape[0])

#             batch_offsets.append(current_offset)
#             current_offset += (max_b + 1)



#         mega_sparse_x = ME.SparseTensor(
#             features=torch.cat(all_feats, dim=0),
#             coordinates=torch.cat(all_coords, dim=0),
#             tensor_stride=x_list[valid_list[0][0]].tensor_stride,
#             device=x_list[valid_list[0][0]].device
#         )



#         encoder_outputs = []

#         for encoder in self.unet.encoders:
#             mega_sparse_x = encoder(mega_sparse_x)
#             encoder_outputs.append(mega_sparse_x)

#         for layer in self.unet.bottleneck:
#             mega_sparse_x = layer(mega_sparse_x)

#         for i, decoder_block in enumerate(self.unet.decoder_blocks):
#             upconv, after_cat = decoder_block

#             mega_sparse_x = upconv(mega_sparse_x)

#             enc_index = len(encoder_outputs) - i - 1
#             if enc_index >= 0 and enc_index < len(encoder_outputs):
#                 mega_sparse_x = ME.cat(mega_sparse_x, encoder_outputs[enc_index])
#                 mega_sparse_x = after_cat(mega_sparse_x)

#         mega_output = self.unet.final_upsample(mega_sparse_x)




#         final_outputs = [None] * len(x_list)
#         for idx_in_valid, (original_idx, _) in enumerate(valid_list):
#             offset = batch_offsets[idx_in_valid]

#             next_offset = batch_offsets[idx_in_valid + 1] if idx_in_valid + 1 < len(batch_offsets) else current_offset




#             final_outputs[original_idx] = self._extract_by_batch_range(mega_output, offset, next_offset)

#         return final_outputs

#     def _extract_by_batch_range(self, sparse_tensor, start_b, end_b):

#         mask = (sparse_tensor.C[:, 0] >= start_b) & (sparse_tensor.C[:, 0] < end_b)
#         new_coords = sparse_tensor.C[mask]

#         return ME.SparseTensor(
#             features=sparse_tensor.F[mask],
#             coordinates=new_coords,
#             tensor_stride=sparse_tensor.tensor_stride,
#             coordinate_manager=sparse_tensor.coordinate_manager,
#             device=sparse_tensor.device
#         )


# from xformers.ops import memory_efficient_attention, LowerTriangularMask

# class FlashVoxelAttention(nn.Module):
#     def __init__(self, dim, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#         self.gamma = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.02)

#     def forward(self, x, coords, mask=None):
#         B, N, C = x.shape
#         H = self.num_heads



#         qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 1, 3, 4)
#         q, k, v = [item.to(target_dtype).contiguous() for item in qkv] # (B, N, H, D)



#         N_aligned = (N + 7) // 8 * 8 
#         


#         bias_aligned = torch.zeros(
#             (B, H, N_aligned, N_aligned), 
#             dtype=target_dtype, 
#             device=x.device
#         )


#         with torch.cuda.amp.autocast(enabled=False):
#             dist_sq = torch.cdist(coords, coords, p=2).pow(2).unsqueeze(1)
#             actual_bias = -self.gamma * dist_sq


#         bias_aligned[:, :, :N, :N] = actual_bias.to(target_dtype)


#         if mask is not None:
#             m_bias = torch.zeros((B, 1, 1, N), device=x.device, dtype=target_dtype)
#             m_bias.masked_fill_(~mask.view(B, 1, 1, N), float("-inf"))
#             bias_aligned[:, :, :N, :N] += m_bias



#         attn_bias = bias_aligned[:, :, :N, :N]



#         out = memory_efficient_attention(
#             q, k, v, 
#             attn_bias=attn_bias
#         )

#         return self.proj(out.to(x.dtype).reshape(B, N, C))


# class SparseHiUNetWithAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, num_blocks=4, use_attention=False, max_levels=4):
#         super().__init__()
#         self.unet = SparseUNetWithAttention(in_channels, out_channels, num_blocks, use_attention)

#         self.max_levels = max_levels
#         # self.level_embedding = torch.nn.Embedding(max_levels+1, in_channels)  # learnable level embedding

#         self.bottleneck_voxel_attention = FlashVoxelAttention(dim=64 * (2 ** 3), num_heads=4)




#     def forward(self, x_list: list[ME.SparseTensor],
#                 attention_on_intermediates: bool = False, voxel_resolution:float = 0.001):

#         valid_list = [(i, x) for i, x in enumerate(x_list) if x is not None and len(x) > 0]
#         if not valid_list: return [None] * len(x_list)


#         all_feats = []
#         all_coords = []
#         batch_offsets = []
#         current_offset = 0
#         max_b_all = 0
#         for _, x in valid_list:
#             coords = x.C.clone()

#             max_b = coords[:, 0].max().item()
#             if max_b+1>max_b_all:
#                 max_b_all = max_b + 1
#             

#             coords[:, 0] += current_offset
#             all_coords.append(coords)
#             all_feats.append(x.F)
#             
#             batch_offsets.append(current_offset)
#             current_offset += (max_b + 1)



#         mega_sparse_x = ME.SparseTensor(
#             features=torch.cat(all_feats, dim=0),
#             coordinates=torch.cat(all_coords, dim=0),
#             tensor_stride=x_list[valid_list[0][0]].tensor_stride,
#             device=x_list[valid_list[0][0]].device
#         )



#         encoder_outputs = []
#         
#         for encoder in self.unet.encoders:
#             mega_sparse_x = encoder(mega_sparse_x)
#             encoder_outputs.append(mega_sparse_x)

#         for layer in self.unet.bottleneck:
#             mega_sparse_x = layer(mega_sparse_x)
#         
#         ### 
#         reverse_batch_indices = []
#         voxel_resolutions = []
#         for idx_in_valid, (original_idx, _) in enumerate(valid_list):
#             offset = batch_offsets[idx_in_valid]
#             next_offset = batch_offsets[idx_in_valid + 1] if idx_in_valid + 1 < len(batch_offsets) else current_offset

#             start_b = offset
#             end_b = next_offset
#             mask = (mega_sparse_x.C[:, 0] >= start_b) & (mega_sparse_x.C[:, 0] < end_b)

#             voxel_resolutions.append(torch.ones(mask.sum(), device=mega_sparse_x.device)*voxel_resolution*(2 ** (self.max_levels-idx_in_valid)))
#             reverse_batch_indices.append(mega_sparse_x.C[mask,0] - offset)
#         voxel_resolutions = torch.concat(voxel_resolutions, dim=0)[..., None]
#         reverse_batch_indices = torch.concat(reverse_batch_indices, dim=0)

#         voxel_coordinate = mega_sparse_x.tensor_stride[0] * mega_sparse_x.C[...,1:].float() * voxel_resolutions + (mega_sparse_x.tensor_stride[0] * voxel_resolutions / 2.0)
#         voxel_feature = mega_sparse_x.F



#         len_b = [(reverse_batch_indices == ost).sum() for ost in range(max_b_all)]
#         actual_batch_size = max_b_all
#         max_pts = max(len_b)
#         device = mega_sparse_x.device

#         if max_pts == 0:

#             pass 
#         else:


#             counts = torch.tensor(len_b, device=device)
#             local_idx = torch.cat([torch.arange(c, device=device) for c in counts]) 
#             

#             attn_in_feat = torch.zeros((actual_batch_size, max_pts, voxel_feature.shape[-1]), device=device)
#             attn_in_coord = torch.zeros((actual_batch_size, max_pts, 3), device=device)
#             attn_mask = torch.zeros((actual_batch_size, max_pts), device=device, dtype=torch.bool)


#             batch_idx_long = reverse_batch_indices.long()
#             attn_in_feat[batch_idx_long, local_idx] = voxel_feature
#             attn_in_coord[batch_idx_long, local_idx] = voxel_coordinate
#             attn_mask[batch_idx_long, local_idx] = True


#             attention_output = self.bottleneck_voxel_attention(attn_in_feat, attn_in_coord, attn_mask)



#             refined_features = attention_output[batch_idx_long, local_idx]

#             mega_sparse_x = ME.SparseTensor(
#                 features=refined_features,
#                 coordinates=mega_sparse_x.C,
#                 tensor_stride=mega_sparse_x.tensor_stride,
#                 coordinate_manager=mega_sparse_x.coordinate_manager
#             )



#         for i, decoder_block in enumerate(self.unet.decoder_blocks):
#             upconv, after_cat = decoder_block
#             
#             mega_sparse_x = upconv(mega_sparse_x)
#             
#             enc_index = len(encoder_outputs) - i - 1
#             if enc_index >= 0 and enc_index < len(encoder_outputs):
#                 mega_sparse_x = ME.cat(mega_sparse_x, encoder_outputs[enc_index])
#                 mega_sparse_x = after_cat(mega_sparse_x)
#         
#         mega_output = self.unet.final_upsample(mega_sparse_x)




#         final_outputs = [None] * len(x_list)
#         for idx_in_valid, (original_idx, _) in enumerate(valid_list):
#             offset = batch_offsets[idx_in_valid]

#             next_offset = batch_offsets[idx_in_valid + 1] if idx_in_valid + 1 < len(batch_offsets) else current_offset
#             



#             final_outputs[original_idx] = self._extract_by_batch_range(mega_output, offset, next_offset)

#         return final_outputs

#     def _extract_by_batch_range(self, sparse_tensor, start_b, end_b):

#         mask = (sparse_tensor.C[:, 0] >= start_b) & (sparse_tensor.C[:, 0] < end_b)
#         new_coords = sparse_tensor.C[mask]

#         return ME.SparseTensor(
#             features=sparse_tensor.F[mask],
#             coordinates=new_coords,
#             tensor_stride=sparse_tensor.tensor_stride,
#             coordinate_manager=sparse_tensor.coordinate_manager,
#             device=sparse_tensor.device
#         ) 
