import MinkowskiEngine as ME
import torch
import torch.nn as nn


class SparseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super().__init__()
        self.expansion = expansion
        mid_channels = out_channels // expansion

        # Main path
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, mid_channels, kernel_size=3, 
            stride=stride, dimension=3
        )
        self.bn1 = ME.MinkowskiBatchNorm(mid_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            mid_channels, out_channels, kernel_size=3, 
            stride=1, dimension=3
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, dimension=3
                ),
                ME.MinkowskiBatchNorm(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class SparseGaussianHead(nn.Module):
    def __init__(self, in_channels=64, out_channels=38):
        """
        Args:
            in_channels: Input channel count (default 64)
            out_channels: Output channel count (number of Gaussian parameters, default 38)
        """
        super().__init__()
        
        self.num_gaussian_parameters = out_channels

        # Sparse 3D convolution network
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
                    ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                except:
                    nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, ME.MinkowskiBatchNorm):
                if hasattr(m, 'bn'):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)

class MultiScaleSparseHead(nn.Module):
    def __init__(self, in_channels=164, base_channels=64, num_blocks=[2, 2, 2, 2], gaussian_out_channels=38):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Initial downsampling layer - 1/2 resolution
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, base_channels, kernel_size=7, 
            stride=2, dimension=3
        )
        self.bn1 = ME.MinkowskiBatchNorm(base_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

        # Four stages of residual blocks, each stage downsamples by 2x
        # Stage 1: 1/4 resolution
        self.stage1 = self._make_stage(
            base_channels, base_channels * 1, num_blocks[0], stride=2
        )
        # Stage 2: 1/8 resolution
        self.stage2 = self._make_stage(
            base_channels * 1, base_channels * 2, num_blocks[1], stride=2
        )
        # Stage 3: 1/16 resolution
        self.stage3 = self._make_stage(
            base_channels * 2, base_channels * 4, num_blocks[2], stride=2
        )
        # Stage 4: 1/32 resolution (but we will upsample back to 1/16)
        self.stage4 = self._make_stage(
            base_channels * 4, base_channels * 8, num_blocks[3], stride=2
        )

        # 1/2 scale output processing
        self.conv_half = ME.MinkowskiConvolution(
            base_channels, base_channels, kernel_size=1, stride=1, dimension=3
        )

        # 1/8 scale output processing
        self.conv_eighth = ME.MinkowskiConvolution(
            base_channels * 2, base_channels, kernel_size=1, stride=1, dimension=3
        )

        # 1/16 scale output processing
        self.conv_sixteenth = ME.MinkowskiConvolution(
            base_channels * 4, base_channels, kernel_size=1, stride=1, dimension=3
        )

        # Upsampling layer for 1/16->1/16 (maintain resolution)
        self.upsample4 = ME.MinkowskiConvolution(
            base_channels * 8, base_channels, kernel_size=3, stride=1, dimension=3
        )

        # Additional skip connection fusion layers
        self.fuse_layers = nn.ModuleList([
            ME.MinkowskiConvolution(
                base_channels * 2, base_channels, kernel_size=1, stride=1, dimension=3
            ) for _ in range(2)
        ])

        # Gaussian parameter conversion heads
        self.gaussian_heads = nn.ModuleList([
            SparseGaussianHead(in_channels=base_channels, out_channels=gaussian_out_channels)
            for _ in range(4)  
        ])
        
        self.init_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """Create a residual stage"""
        blocks = []

        blocks.append(SparseResidualBlock(in_channels, out_channels, stride))
       
        for _ in range(1, num_blocks):
            blocks.append(SparseResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)
    
    def forward(self, x: ME.SparseTensor):

        # 1/2 resolution features
        x_half = self.conv1(x)
        x_half = self.bn1(x_half)
        x_half = self.relu(x_half)

        # 1/4 resolution features
        x_quarter = self.stage1(x_half)

        # 1/8 resolution features
        x_eighth = self.stage2(x_quarter)

        # 1/16 resolution features
        x_sixteenth = self.stage3(x_eighth)

        # 1/32 resolution features
        x_thirtysecond = self.stage4(x_sixteenth)
        # Upsample back to 1/16 equivalent resolution
        x_sixteenth2 = self.upsample4(x_thirtysecond)

        # 1/8 resolution feature processing
        x_eighth_proc = self.conv_eighth(x_eighth)

        # 1/16 resolution feature fusion
        # First adjust x_sixteenth channel count to 64
        x_sixteenth_adjusted = self.conv_sixteenth(x_sixteenth)
        # Then perform addition
        x_sixteenth_final = x_sixteenth_adjusted + x_sixteenth2

        # Create multi-scale feature list
        features = [
            self.conv_half(x_half),    # 1/2
            x_quarter,                  # 1/4
            x_eighth_proc,              # 1/8
            x_sixteenth_final           # 1/16
        ]
        
    
        gaussian_outputs = []
        for i, feat in enumerate(features):
            gaussian_outputs.append(self.gaussian_heads[i](feat))
        
        return gaussian_outputs
    
    def init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                try:
                    ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                except:
                    nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, ME.MinkowskiBatchNorm):
                if hasattr(m, 'bn'):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
            
            elif isinstance(m, ME.MinkowskiConvolutionTranspose):
                try:
                    ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                except:
                    nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)



