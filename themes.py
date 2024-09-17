from __future__ import annotations
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

class IndonesiaTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.red,
        secondary_hue: colors.Color | str = colors.gray,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="linear-gradient(to bottom, #e0e0e0, #7d7d7d)",  # Gradasi abu-abu muda ke abu-abu tua
            body_background_fill_dark="linear-gradient(to bottom, #7d7d7d, #4a4a4a)",  # Gradasi abu-abu tua ke lebih gelap untuk dark mode
            button_primary_background_fill="linear-gradient(90deg, #d84a4a, #b33030)",  # Merah ke merah tua
            button_primary_background_fill_hover="linear-gradient(90deg, #e85b5b, #cc4b4b)",  # Merah lebih terang untuk hover
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, #b33030, #8f1f1f)",  # Merah tua untuk dark mode
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )
