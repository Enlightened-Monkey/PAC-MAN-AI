from __future__ import annotations

import numpy as np
import cv2

from src.models.segmentation_detector import (
    build_class_layers,
    build_group_layers,
    detect_playfield_bbox,
    detect_lives_icons,
    extract_instances,
    parse_hud_numbers,
)


def test_extract_instances_returns_components():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 3:7] = 6  # pacman
    mask[10:15, 10:14] = 7  # blinky

    id_to_class = {0: "empty", 6: "pacman", 7: "blinky"}
    instances = extract_instances(mask, id_to_class, min_area=4)

    labels = sorted(obj["label"] for obj in instances)
    assert labels == ["blinky", "pacman"]
    assert all(len(obj["bbox"]) == 4 for obj in instances)


def test_layers_are_built_for_classes_and_groups():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 6  # pacman
    mask[4:7, 4:7] = 7  # blinky
    class_to_id = {
        "empty": 0,
        "pacman": 6,
        "blinky": 7,
        "pinky": 8,
        "inky": 9,
        "clyde": 10,
        "frightened_ghost": 11,
        "ghost_eyes": 12,
        "fruit": 13,
        "pellet": 2,
        "power_pellet": 3,
        "wall": 1,
        "ghost_door": 4,
        "ghost_house": 5,
    }

    class_layers = build_class_layers(mask, class_to_id)
    group_layers = build_group_layers(mask, class_to_id)

    assert class_layers["pacman"].sum() > 0
    assert class_layers["blinky"].sum() > 0
    assert group_layers["pacman"].sum() > 0
    assert group_layers["ghosts"].sum() > 0


def test_detect_lives_icons_finds_yellow_hud_blobs():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[100:108, 10:18] = np.array([255, 255, 0], dtype=np.uint8)
    image[102:110, 28:36] = np.array([255, 255, 0], dtype=np.uint8)

    icons = detect_lives_icons(image)

    assert len(icons) >= 2
    assert all(icon["label"] == "life_icon" for icon in icons)


def test_detect_playfield_bbox_finds_blue_maze_region():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[40:170, 30:260] = np.array([0, 0, 255], dtype=np.uint8)

    x, y, w, h = detect_playfield_bbox(image)

    assert x <= 30
    assert y <= 40
    assert x + w >= 260
    assert y + h >= 170


def test_parse_hud_numbers_reads_left_and_right_groups():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.putText(image, "123", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(image, "456", (180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    hud = parse_hud_numbers(image)

    assert hud["score_text"] is not None
    assert hud["high_score_text"] is not None
    assert hud["digit_count"] >= 4
