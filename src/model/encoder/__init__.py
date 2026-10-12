from typing import Optional

from .encoder import Encoder

from .encoder_volodsplat import EncoderVoLoDSplat, EncoderVoLoDSplatCfg

from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_volodsplat import EncoderVisualizerVoLoDSplat
ENCODERS = {
    "volodsplat": (EncoderVoLoDSplat, EncoderVisualizerVoLoDSplat),
}

EncoderCfg = EncoderVoLoDSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
