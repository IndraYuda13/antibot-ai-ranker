from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORD_UNIVERSES: list[dict[str, str]] = [
    {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"},
    {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"},
    {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X"},
    {"2-1": "1", "1+1": "2", "1+2": "3", "2+2": "4", "3+2": "5", "2+4": "6", "3+4": "7", "4+4": "8", "1+8": "9", "5+6": "11"},
    {"zoo": "200", "ozo": "020", "ooz": "002", "soo": "500", "oso": "050", "oos": "005", "lol": "101", "sos": "505", "zoz": "202", "lll": "111"},
    {"ice": "1c3", "pan": "p@n", "tea": "t3@", "top": "t0p", "toy": "t0y", "tap": "t@p", "sky": "5ky", "day": "d@y", "wet": "w3t", "hot": "h0t"},
    {"cat": "c@t", "dog": "d0g", "fox": "f0x", "cow": "c0w", "ant": "@nt", "crab": "cr@b", "deer": "d33r", "duck": "dvck", "panda": "p@nd@", "mouse": "m0use"},
]


@dataclass(frozen=True)
class SyntheticConfig:
    option_count: int = 3
    seed: int = 1337
    output_dir: Path = Path("data/synthetic")
    noise: bool = True
    dark_theme: bool = False
    image_format: str = "png"


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _rng_for(cfg: SyntheticConfig, challenge_id: str) -> random.Random:
    return random.Random(f"{cfg.seed}:{challenge_id}")


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _color(rng: random.Random, *, dark_theme: bool, shadow: bool) -> tuple[int, int, int]:
    if dark_theme:
        lo, hi = (1, 80) if shadow else (214, 254)
    else:
        lo, hi = (174, 254) if shadow else (1, 80)
    return tuple(rng.randint(lo, hi) for _ in range(3))


def _draw_text_image(text: str, path: Path, rng: random.Random, *, cfg: SyntheticConfig) -> dict[str, object]:
    font_size = rng.randint(20, 25)
    font = _font(font_size)
    width = max(50, (len(text) + 1) * (12 if len(text) >= 10 else 15))
    height = 42
    bg = (18, 18, 18) if cfg.dark_theme else (255, 255, 255)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    angle = rng.randint(-2, 2) if width > 125 else rng.randint(-8, 8)
    text_color = _color(rng, dark_theme=cfg.dark_theme, shadow=False)
    shadow_color = _color(rng, dark_theme=cfg.dark_theme, shadow=True)

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    bbox = layer_draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = (height - (bbox[3] - bbox[1])) // 2 - 1
    layer_draw.text((x + 1, y + 1), text, font=font, fill=(*shadow_color, 255))
    layer_draw.text((x, y), text, font=font, fill=(*text_color, 255))
    rotated = layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(width // 2, height // 2))
    image = Image.alpha_composite(image.convert("RGBA"), rotated).convert("RGB")
    draw = ImageDraw.Draw(image)
    if cfg.noise:
        for _ in range(max(1, round(width / rng.randint(16, 22) * 10))):
            x1 = rng.randint(1, width - 3)
            y1 = rng.randint(1, height - 3)
            draw.line((x1, y1, x1 + rng.randint(1, 2), y1 + rng.choice([-1, 1])), fill=text_color)
        if rng.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.choice([0.2, 0.35, 0.5])))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {"width": width, "height": height, "angle": angle, "font_size": font_size, "noise": cfg.noise, "dark_theme": cfg.dark_theme}


def _sample_universe(rng: random.Random, option_count: int) -> tuple[list[str], dict[str, str], bool]:
    if option_count < 2:
        raise ValueError("option_count must be >= 2")
    universe = dict(rng.choice(WORD_UNIVERSES))
    if option_count > len(universe):
        raise ValueError(f"option_count {option_count} exceeds word universe size {len(universe)}")
    keys = rng.sample(list(universe.keys()), option_count)
    pairs = {key: universe[key] for key in keys}
    flipped = bool(rng.randint(0, 1))
    if flipped:
        pairs = {value: key for key, value in pairs.items()}
    return list(pairs.keys()), pairs, flipped


def generate_example(challenge_id: str, cfg: SyntheticConfig) -> dict[str, object]:
    rng = _rng_for(cfg, challenge_id)
    ordered_tokens, pairs, flipped = _sample_universe(rng, cfg.option_count)
    option_ids = [str(rng.randint(1000, 9999)) for _ in ordered_tokens]
    while len(set(option_ids)) < len(option_ids):
        option_ids = [str(rng.randint(1000, 9999)) for _ in ordered_tokens]
    ordered_options = list(zip(option_ids, [pairs[token] for token in ordered_tokens]))
    shuffled_options = ordered_options[:]
    rng.shuffle(shuffled_options)

    base = Path("images") / _safe_id(challenge_id)
    question_rel = base / f"question.{cfg.image_format}"
    metadata = {
        "flipped": flipped,
        "question": _draw_text_image(", ".join(ordered_tokens), cfg.output_dir / question_rel, rng, cfg=cfg),
        "options": {},
    }
    option_texts: dict[str, str] = {}
    option_images: dict[str, str] = {}
    for option_id, text in shuffled_options:
        rel = base / f"option_{option_id}.{cfg.image_format}"
        metadata["options"][option_id] = _draw_text_image(text, cfg.output_dir / rel, rng, cfg=cfg)
        option_texts[option_id] = text
        option_images[option_id] = str(rel)

    return {
        "challenge_id": challenge_id,
        "source": "synthetic_antibotlinks_v1",
        "option_count": cfg.option_count,
        "question_text": ", ".join(ordered_tokens),
        "option_texts": option_texts,
        "correct_order": option_ids,
        "images": {"question": str(question_rel), "options": option_images},
        "metadata": metadata,
    }


def generate_dataset(*, count: int, cfg: SyntheticConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / f"synthetic_{cfg.option_count}opt_{count}.jsonl"
    with out.open("w") as fh:
        for idx in range(1, count + 1):
            challenge_id = f"synthetic_{cfg.option_count}opt_{idx:06d}"
            fh.write(json.dumps(generate_example(challenge_id, cfg), ensure_ascii=False) + "\n")
    return out
