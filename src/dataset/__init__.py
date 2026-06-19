__all__ = [
    "DEFAULT_GENERAL_SPRITE_SHEET",
    "PacmanMapDatasetGenerator",
    "PacmanSpriteSheetExtractor",
]


def __getattr__(name: str):
    if name in __all__:
        from src.dataset import pacman_map_dataset

        return getattr(pacman_map_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")