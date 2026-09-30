"""Composable FFmpeg video filters."""

from __future__ import annotations


def audio_distortion_filter() -> str:
    # TV-style anonymisation: pitch is raised slightly, timing is compensated
    # back into sync, and the spectrum is narrowed to reduce voice naturalness.
    return (
        "aresample=48000,"
        "asetrate=48000*1.08,"
        "aresample=48000,"
        "atempo=0.925926,"
        "highpass=f=120,"
        "lowpass=f=6500,"
        "acompressor=threshold=0.08:ratio=2.5:attack=20:release=250:makeup=1.5,"
        "alimiter=limit=0.95"
    )


def line_drawing_filter() -> str:
    # Cartoon-like black ink on white paper. The contrast step improves edge
    # detection, while negate turns FFmpeg's bright-edge output into dark lines.
    return "format=gray,eq=contrast=1.35:brightness=-0.01,edgedetect=low=0.025:high=0.11,negate,eq=contrast=1.15:brightness=0.03"


def detailed_line_drawing_filter() -> str:
    # A lighter cartoon treatment for review work where more facial and scene
    # detail should remain visible. This is less anonymising than the standard
    # cartoon mode because the lower thresholds preserve finer edges.
    return "format=gray,eq=contrast=1.20:brightness=0.00,edgedetect=low=0.012:high=0.055,negate,eq=contrast=1.05:brightness=0.04"


def pixelation_filter() -> str:
    return "scale=iw/20:ih/20,scale=iw*20:ih*20:flags=neighbor,setsar=1"


def blur_filter() -> str:
    return "gblur=sigma=20"


def silhouette_filter() -> str:
    # A true person-mask silhouette needs segmentation. This local-only FFmpeg
    # recipe deliberately reduces the whole frame into coarse high-contrast
    # shapes, preserving motion while discarding detail.
    return "format=gray,gblur=sigma=10,eq=contrast=3.0:brightness=-0.22,curves=all='0/0 0.48/0 0.62/1 1/1',format=gray"


def grayscale_filter() -> str:
    return "format=gray"


def resize_filter(height: int) -> str:
    return f"scale=-2:{height}"
