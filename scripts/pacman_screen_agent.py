#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
import select
import shutil
import socketserver
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

# OpenCV wheel in this env ships Qt xcb platform plugin but no wayland plugin.
# Force xcb on Wayland sessions so preview window can be created.
if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2
import mss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.pacman_map_dataset import CLASS_TO_ID
from src.models.segmentation_detector import (
    ID_TO_CLASS,
    SegmentationDetector,
    detect_playfield_bbox,
    extract_instances,
    remap_instances_to_image,
)

try:
    from pynput.keyboard import Controller as KeyboardController
    from pynput.keyboard import Key
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing keyboard control dependency. Install with: pip install pynput"
    ) from exc


GHOST_NAMES = ("blinky", "pinky", "inky", "clyde", "frightened_ghost", "ghost_eyes")

YOLO_COLORS = {
    "pacman": (0, 215, 255),
    "blinky": (80, 80, 255),
    "pinky": (255, 120, 255),
    "inky": (255, 220, 120),
    "clyde": (50, 170, 255),
    "frightened_ghost": (255, 180, 50),
    "ghost_eyes": (220, 220, 220),
    "pellet": (120, 255, 180),
    "power_pellet": (80, 255, 120),
    "fruit": (70, 70, 255),
}


@dataclass
class AgentState:
    enabled: bool = False
    stop_requested: bool = False
    last_dir: str | None = None
    detection_enabled: bool = True
    locked_roi: tuple[int, int, int, int] | None = None
    last_playfield_bbox: tuple[int, int, int, int] | None = None


@dataclass
class VisionMemory:
    """Vision-only latent state inferred from frame-to-frame observations."""

    frame_idx: int = 0
    level_estimate: int = 1
    frightened_timer_est: int = 0
    steps_since_pellet_est: int = 0
    last_pellet_count: int = -1
    last_power_count: int = -1
    last_power_positions: list[tuple[int, int]] = field(default_factory=list)
    last_pacman_pos: tuple[int, int] | None = None
    last_event: str = "init"


# Approximate frightened duration schedule for fair, vision-only internal timing.
FRIGHTENED_TICKS_BY_LEVEL: dict[int, int] = {
    1: 60, 2: 40, 3: 30, 4: 20, 5: 20, 6: 50, 7: 20, 8: 20,
    9: 10, 10: 50, 11: 20, 12: 10, 13: 10, 14: 10, 15: 10,
    16: 10, 17: 20, 18: 10,
}


class MjpegPreviewServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._frame_lock = threading.Lock()
        self._frame_jpeg: bytes | None = None
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        owner = self

        class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    body = (
                        "<html><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif;'>"
                        "<div style='padding:8px'>PAC-MAN Live Preview</div>"
                        "<img src='/stream.mjpg' style='display:block;max-width:100vw;max-height:92vh;margin:auto'/>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != "/stream.mjpg":
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                while True:
                    with owner._frame_lock:
                        frame = owner._frame_jpeg
                    if frame is None:
                        time.sleep(0.03)
                        continue

                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                    except (BrokenPipeError, ConnectionResetError):
                        break

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ReusableThreadingTCPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update_frame(self, rgb_frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._frame_lock:
            self._frame_jpeg = encoded.tobytes()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class TerminalToggle:
    def __init__(self, state: AgentState) -> None:
        self.state = state
        self._thread: threading.Thread | None = None
        self._old_settings = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _reader_loop(self) -> None:
        while not self.state.stop_requested:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key.lower() == "s":
                self.state.enabled = not self.state.enabled
                status = "ON" if self.state.enabled else "OFF"
                print(f"\n[control] auto-input: {status}")
            elif key.lower() == "d":
                self.state.detection_enabled = not self.state.detection_enabled
                status = "ON" if self.state.detection_enabled else "OFF"
                print(f"\n[control] detection: {status}")
            elif key.lower() == "r":
                if self.state.locked_roi is None:
                    if self.state.last_playfield_bbox is not None:
                        self.state.locked_roi = self.state.last_playfield_bbox
                        x, y, w, h = self.state.locked_roi
                        print(f"\n[control] roi locked to last bbox: ({x},{y},{w},{h})")
                    else:
                        print("\n[control] roi lock requested, but no bbox detected yet")
                else:
                    self.state.locked_roi = None
                    print("\n[control] roi unlocked")
            elif key.lower() == "q":
                self.state.stop_requested = True
                print("\n[control] quit requested")


def centroid(binary_mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def is_passable(walls: np.ndarray, x: int, y: int, radius: int = 3) -> bool:
    h, w = walls.shape
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius + 1)
    y1 = min(h, y + radius + 1)
    patch = walls[y0:y1, x0:x1]
    return float(patch.mean()) < 0.35


def nearest_ghost_distance(ghost_map: np.ndarray, x: int, y: int) -> float:
    ys, xs = np.where(ghost_map)
    if len(xs) == 0:
        return 9999.0
    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    return float(np.sqrt(dx * dx + dy * dy).min())


def _component_centers(binary: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return []
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    out: list[tuple[int, int]] = []
    for i in range(1, n):
        _x, _y, _w, _h, area = stats[i]
        if int(area) <= 0:
            continue
        cx, cy = centroids[i]
        out.append((int(cx), int(cy)))
    return out


def _any_near(ref: tuple[int, int], pts: list[tuple[int, int]], radius: float) -> bool:
    rx, ry = ref
    rr2 = float(radius * radius)
    for px, py in pts:
        dx = float(px - rx)
        dy = float(py - ry)
        if dx * dx + dy * dy <= rr2:
            return True
    return False


def update_vision_memory(memory: VisionMemory, mask: np.ndarray, class_to_id: dict[str, int]) -> VisionMemory:
    memory.frame_idx += 1

    pellet_id = class_to_id.get("pellet")
    power_id = class_to_id.get("power_pellet")
    pac_id = class_to_id.get("pacman")
    fr_ghost_id = class_to_id.get("frightened_ghost")

    pellet_count = int((mask == pellet_id).sum()) if pellet_id is not None else 0
    power_count = int((mask == power_id).sum()) if power_id is not None else 0
    total_collectibles = pellet_count + power_count

    pac = centroid(mask == pac_id) if pac_id is not None else None
    frightened_visible = bool(fr_ghost_id is not None and (mask == fr_ghost_id).any())
    power_positions = _component_centers(mask == power_id) if power_id is not None else []

    # Pellet timing estimate from visual change in collectible count.
    if memory.last_pellet_count >= 0 and memory.last_power_count >= 0:
        prev_total = memory.last_pellet_count + memory.last_power_count
        if total_collectibles < prev_total:
            memory.steps_since_pellet_est = 0
            memory.last_event = "pellet_eaten"
        else:
            memory.steps_since_pellet_est += 1
    else:
        memory.steps_since_pellet_est = 0

    # Level estimate (vision-only): collectible map reset indicates new board.
    if memory.last_pellet_count >= 0 and memory.last_power_count >= 0:
        prev_total = memory.last_pellet_count + memory.last_power_count
        if total_collectibles - prev_total > 40:
            memory.level_estimate = min(memory.level_estimate + 1, 21)
            memory.last_event = "level_reset_detected"

    # Power-pellet trigger estimate: power components dropped and Pac-Man was near previous power slot.
    power_triggered = False
    if (
        memory.last_power_count >= 0
        and power_count < memory.last_power_count
        and pac is not None
        and memory.last_power_positions
        and _any_near(pac, memory.last_power_positions, radius=14.0)
    ):
        frightened_ticks = FRIGHTENED_TICKS_BY_LEVEL.get(memory.level_estimate, 0)
        memory.frightened_timer_est = frightened_ticks
        memory.last_event = "power_pellet_eaten"
        power_triggered = True

    # Keep timer alive while frightened ghosts are visible, otherwise decay.
    if frightened_visible:
        memory.frightened_timer_est = max(memory.frightened_timer_est, 2)
    elif memory.frightened_timer_est > 0 and not power_triggered:
        memory.frightened_timer_est -= 1

    memory.last_pellet_count = pellet_count
    memory.last_power_count = power_count
    memory.last_power_positions = power_positions
    memory.last_pacman_pos = pac
    return memory


def choose_direction(mask: np.ndarray, class_to_id: dict[str, int], state: AgentState, memory: VisionMemory) -> str | None:
    pac_id = class_to_id.get("pacman")
    wall_id = class_to_id.get("wall")
    pellet_id = class_to_id.get("pellet")
    power_id = class_to_id.get("power_pellet")

    if pac_id is None or wall_id is None:
        return None

    pac = centroid(mask == pac_id)
    if pac is None:
        return None

    px, py = pac
    walls = mask == wall_id
    pellets = (mask == pellet_id) | (mask == power_id if power_id is not None else False)

    ghost_map = np.zeros_like(mask, dtype=bool)
    frightened_map = np.zeros_like(mask, dtype=bool)
    for name in GHOST_NAMES:
        cid = class_to_id.get(name)
        if cid is not None:
            if name == "frightened_ghost":
                frightened_map |= mask == cid
            elif name == "ghost_eyes":
                continue
            else:
                ghost_map |= mask == cid

    directions = {
        "up": (0, -1, Key.up),
        "down": (0, 1, Key.down),
        "left": (-1, 0, Key.left),
        "right": (1, 0, Key.right),
    }

    best_name = None
    best_score = -1e9

    for name, (dx, dy, _) in directions.items():
        nx = px + dx * 5
        ny = py + dy * 5
        if not is_passable(walls, nx, ny):
            continue

        h, w = mask.shape
        bx0 = max(0, nx - 8)
        by0 = max(0, ny - 8)
        bx1 = min(w, nx + 9)
        by1 = min(h, ny + 9)
        pellet_score = float(pellets[by0:by1, bx0:bx1].sum())

        gdist = nearest_ghost_distance(ghost_map, nx, ny)
        ghost_penalty = 18.0 / (gdist + 1.0)

        # Vision-only frightened timer helps switch from avoid -> chase behavior.
        frightened_bonus = 0.0
        if memory.frightened_timer_est > 0:
            fgdist = nearest_ghost_distance(frightened_map, nx, ny)
            frightened_bonus = 12.0 / (fgdist + 1.0)
            ghost_penalty *= 0.35

        # If agent has not observed pellet gain for long, bias toward exploration.
        explore_bonus = min(memory.steps_since_pellet_est / 80.0, 1.0)

        forward_bonus = 0.8 if state.last_dir == name else 0.0
        score = pellet_score * 0.35 - ghost_penalty + frightened_bonus + forward_bonus + explore_bonus

        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def direction_to_key(direction: str) -> Key:
    return {
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
    }[direction]


def color_for_label(label: str) -> tuple[int, int, int]:
    return YOLO_COLORS.get(label, (230, 230, 230))


def draw_yolo_style_overlay(
    rgb: np.ndarray,
    instances: list[dict],
    playfield_bbox: tuple[int, int, int, int],
    direction: str | None,
    auto_enabled: bool,
    fps: float,
    memory: VisionMemory | None = None,
) -> np.ndarray:
    vis = rgb.copy()
    x0, y0, bw, bh = playfield_bbox

    # Playfield outline.
    cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 255), 2)

    # YOLO-like detections with label chips.
    for obj in instances:
        x, y, w, h = [int(v) for v in obj["bbox"]]
        label = str(obj.get("label", "obj"))
        area = int(obj.get("area", max(1, w * h)))
        conf = min(0.99, max(0.25, 0.35 + min(area / 1200.0, 0.6)))
        color = color_for_label(label)

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(0, y - th - 8)
        cv2.rectangle(vis, (x, ty), (x + tw + 8, ty + th + 8), color, -1)
        cv2.putText(
            vis,
            tag,
            (x + 4, ty + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    # HUD panel.
    panel_h = 58
    cv2.rectangle(vis, (0, 0), (540, panel_h), (15, 15, 15), -1)
    status = "ON" if auto_enabled else "OFF"
    cv2.putText(vis, f"AUTO: {status}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"DIR: {direction or '-'}",
        (170, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"FPS: {fps:.1f}   DET: {len(instances)}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    if memory is not None:
        cv2.putText(
            vis,
            f"lvl~{memory.level_estimate}  fr_t~{memory.frightened_timer_est:02d}  pellet_t~{memory.steps_since_pellet_est:02d}",
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 220, 170),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"event: {memory.last_event}",
            (380, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        "[s] toggle auto   [q] quit",
        (285, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return vis


def resolve_model_path(model_arg: str) -> Path:
    requested = Path(model_arg).expanduser()
    candidates: list[Path] = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append((Path.cwd() / requested).resolve())
        candidates.append((PROJECT_ROOT / requested).resolve())

    # Common fallback paths used in this repository.
    candidates.extend(
        [
            (PROJECT_ROOT / "models" / requested.name).resolve(),
            (PROJECT_ROOT / "notebooks" / "models" / requested.name).resolve(),
            (PROJECT_ROOT / "models" / "segmentation_unet.pt").resolve(),
            (PROJECT_ROOT / "models" / "segmentation_unet_long.pt").resolve(),
            (PROJECT_ROOT / "notebooks" / "models" / "segmentation_unet_combined.pt").resolve(),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    searched = "\n - ".join(str(p) for p in seen)
    raise FileNotFoundError(
        "Model checkpoint not found. Checked:\n"
        f" - {searched}\n"
        "Pass --model with an existing .pt checkpoint path."
    )


def _is_black_frame(rgb: np.ndarray) -> bool:
    return int(rgb.max()) == 0


def _capture_with_spectacle(output_path: Path) -> np.ndarray:
    cmd = ["spectacle", "-b", "-n", "-o", str(output_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bgr = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"spectacle produced unreadable file: {output_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_roi_arg(raw: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(p.strip()) for p in raw.split(",")]
    except ValueError as exc:
        raise ValueError("ROI must be integers in format x,y,w,h") from exc
    if len(parts) != 4:
        raise ValueError("ROI must contain 4 values: x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be > 0")
    return (x, y, w, h)


def clamp_bbox(bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    x2 = max(x + 1, min(frame_w, x + w))
    y2 = max(y + 1, min(frame_h, y + h))
    return (x, y, x2 - x, y2 - y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime PAC-MAN screen agent: screen capture + segmentation + arrow key control."
    )
    parser.add_argument(
        "--model",
        default="models/segmentation_unet_combined.pt",
        help="Path to trained segmentation checkpoint (.pt)",
    )
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index for mss (default: 1)")
    parser.add_argument("--roi", type=str, default=None, help="Fixed ROI as x,y,w,h")
    parser.add_argument(
        "--capture-backend",
        choices=["auto", "mss", "spectacle"],
        default="auto",
        help="Screen capture backend (default: auto)",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Inference loop FPS")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV preview window")
    parser.add_argument("--preview-web", action="store_true", help="Serve live preview in browser via MJPEG")
    parser.add_argument("--preview-port", type=int, default=8765, help="Port for web preview (default: 8765)")
    parser.add_argument("--dry-run", action="store_true", help="Run detection without sending key presses")
    parser.add_argument("--list-monitors", action="store_true", help="Print available monitors and exit")
    args = parser.parse_args()

    fixed_roi = parse_roi_arg(args.roi) if args.roi else None

    if args.list_monitors:
        with mss.MSS() as sct:
            print("[agent] Available monitors:")
            for idx, mon in enumerate(sct.monitors):
                print(
                    f"  {idx}: left={mon['left']} top={mon['top']} "
                    f"width={mon['width']} height={mon['height']}"
                )
        return

    model_path = resolve_model_path(args.model)

    detector = SegmentationDetector.load(model_path, device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    keyboard = KeyboardController()

    web_preview = False
    if args.preview_web:
        web_preview = True
    elif args.preview and sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
        # On some Wayland setups, OpenCV windows are created but not visible to the user.
        web_preview = True

    web_server: MjpegPreviewServer | None = None
    web_port = int(args.preview_port)
    if web_preview:
        last_error: Exception | None = None
        for offset in range(20):
            candidate_port = web_port + offset
            try:
                web_server = MjpegPreviewServer("127.0.0.1", candidate_port)
                web_server.start()
                web_port = candidate_port
                break
            except OSError as exc:
                last_error = exc
                web_server = None
        if web_server is None:
            raise RuntimeError(
                f"Could not start web preview on ports {web_port}-{web_port + 19}: {last_error}"
            ) from last_error

    state = AgentState(enabled=False, stop_requested=False, last_dir=None)
    memory = VisionMemory()
    if fixed_roi is not None:
        state.locked_roi = fixed_roi
    toggle = TerminalToggle(state)
    toggle.start()

    print("[agent] Started")
    print("[agent] Controls: s=auto ON/OFF, d=detection ON/OFF, r=lock/unlock ROI, q=quit")
    print(f"[agent] Model: {model_path}")
    if args.preview and sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
        print("[agent] Wayland detected: forcing Qt backend to xcb for preview window")
    if web_server is not None:
        print(f"[agent] Web preview: http://127.0.0.1:{web_port}")
    if state.locked_roi is not None:
        x, y, w, h = state.locked_roi
        print(f"[agent] Fixed ROI: ({x},{y},{w},{h})")

    frame_dt = 1.0 / max(args.fps, 0.1)
    last_log = 0.0
    last_tick = time.time()

    capture_backend = args.capture_backend
    use_wayland = bool(sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"))
    spectacle_path = shutil.which("spectacle")
    spectacle_tmp = PROJECT_ROOT / "data" / "pellet_probe" / "_spectacle_capture.png"
    spectacle_tmp.parent.mkdir(parents=True, exist_ok=True)

    try:
        with mss.MSS() as sct:
            monitors = sct.monitors
            if args.monitor >= len(monitors):
                raise ValueError(f"Monitor index {args.monitor} not found. Available 1..{len(monitors)-1}")

            mon = monitors[args.monitor]

            if capture_backend == "auto":
                probe = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]
                probe_rgb = cv2.cvtColor(probe, cv2.COLOR_BGR2RGB)
                if use_wayland and _is_black_frame(probe_rgb):
                    if spectacle_path:
                        capture_backend = "spectacle"
                        print("[agent] mss frame is black on Wayland, switching capture backend to spectacle")
                    else:
                        raise RuntimeError(
                            "Wayland session blocks mss capture (black frames). "
                            "Install spectacle or run an X11 session, then retry."
                        )
                else:
                    capture_backend = "mss"

            if capture_backend == "spectacle" and not spectacle_path:
                raise RuntimeError("Capture backend spectacle requested, but spectacle is not installed.")

            while not state.stop_requested:
                t0 = time.time()

                if capture_backend == "spectacle":
                    rgb = _capture_with_spectacle(spectacle_tmp)
                else:
                    raw = np.array(sct.grab(mon), dtype=np.uint8)
                    bgr = raw[:, :, :3]
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                frame_h, frame_w = rgb.shape[:2]
                search_bbox = (0, 0, frame_w, frame_h)
                if state.locked_roi is not None:
                    search_bbox = clamp_bbox(state.locked_roi, frame_w, frame_h)

                sx, sy, sw, sh = search_bbox
                search_rgb = rgb[sy : sy + sh, sx : sx + sw]

                direction = None
                instances: list[dict] = []
                x0, y0, bw, bh = search_bbox
                if state.detection_enabled:
                    local_x, local_y, local_w, local_h = detect_playfield_bbox(search_rgb)
                    x0, y0, bw, bh = (sx + local_x, sy + local_y, local_w, local_h)
                    x0, y0, bw, bh = clamp_bbox((x0, y0, bw, bh), frame_w, frame_h)
                    state.last_playfield_bbox = (x0, y0, bw, bh)

                    crop = rgb[y0 : y0 + bh, x0 : x0 + bw]
                    if crop.size > 0:
                        resized = cv2.resize(crop, (224, 248), interpolation=cv2.INTER_AREA)
                        mask = detector.predict_mask(resized)

                        memory = update_vision_memory(memory, mask, CLASS_TO_ID)
                        direction = choose_direction(mask, CLASS_TO_ID, state, memory)
                        instances_local = extract_instances(mask, ID_TO_CLASS, min_area=10)
                        instances = remap_instances_to_image(instances_local, (x0, y0, bw, bh), mask.shape)
                        if direction is not None:
                            state.last_dir = direction
                            if state.enabled and not args.dry_run:
                                key = direction_to_key(direction)
                                keyboard.press(key)
                                keyboard.release(key)

                if args.preview:
                    now = time.time()
                    fps = 1.0 / max(1e-6, now - last_tick)
                    last_tick = now
                    vis = draw_yolo_style_overlay(
                        rgb, instances, (x0, y0, bw, bh), direction, state.enabled, fps, memory=memory
                    )
                    if state.locked_roi is not None:
                        rx, ry, rw, rh = clamp_bbox(state.locked_roi, frame_w, frame_h)
                        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 180, 0), 2)
                        cv2.putText(vis, "ROI", (rx + 4, max(14, ry - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)
                    if not state.detection_enabled:
                        cv2.putText(vis, "DETECTION: OFF", (560, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 120, 120), 2, cv2.LINE_AA)
                    if web_server is not None:
                        web_server.update_frame(vis)
                    cv2.imshow("PAC-MAN Screen Agent", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        state.stop_requested = True
                elif web_server is not None:
                    now = time.time()
                    fps = 1.0 / max(1e-6, now - last_tick)
                    last_tick = now
                    vis = draw_yolo_style_overlay(
                        rgb, instances, (x0, y0, bw, bh), direction, state.enabled, fps, memory=memory
                    )
                    if state.locked_roi is not None:
                        rx, ry, rw, rh = clamp_bbox(state.locked_roi, frame_w, frame_h)
                        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 180, 0), 2)
                        cv2.putText(vis, "ROI", (rx + 4, max(14, ry - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)
                    if not state.detection_enabled:
                        cv2.putText(vis, "DETECTION: OFF", (560, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 120, 120), 2, cv2.LINE_AA)
                    web_server.update_frame(vis)

                now = time.time()
                if now - last_log > 1.0:
                    print(
                        f"[agent] auto={'ON' if state.enabled else 'OFF'} det={'ON' if state.detection_enabled else 'OFF'} dir={direction} "
                        f"bbox=({x0},{y0},{bw},{bh}) fr_t~{memory.frightened_timer_est} pellet_t~{memory.steps_since_pellet_est} evt={memory.last_event}"
                    )
                    last_log = now

                elapsed = time.time() - t0
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
    finally:
        if web_server is not None:
            web_server.close()
        toggle.close()
        cv2.destroyAllWindows()
        print("[agent] Stopped")


if __name__ == "__main__":
    main()
