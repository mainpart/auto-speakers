#!/usr/bin/env python3
"""
spkreg — реестр голосовых отпечатков для whispermlx --speaker_embeddings.

Хранит эмбеддинги спикеров между запусками, чтобы SPEAKER_00 из нового файла
можно было сопоставить с уже размеченным человеком.

Работает поверх JSON, который отдаёт:
    whispermlx audio.wav --diarize --speaker_embeddings -f json

Реестр: ~/.local/share/spkreg/registry.json
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = Path(
    os.environ.get("SPKREG_HOME", Path.home() / ".local/share/spkreg")
) / "registry.json"

# Порог косинусной близости для эмбеддингов pyannote community-1 (256 измерений).
# Замер на реальной записи: один человек в двух независимых прогонах — 0.990-0.992,
# два разных человека — 0.439-0.442. Порог посередине, с запасом в обе стороны.
DEFAULT_THRESHOLD = 0.70


# --------------------------------------------------------------------------- io

def load_registry() -> dict:
    if not REGISTRY.exists():
        return {"version": 1, "speakers": {}}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def save_registry(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "speaker_embeddings" not in data:
        sys.exit(
            f"{path}: нет поля speaker_embeddings.\n"
            "Прогони whispermlx с флагами --diarize --speaker_embeddings"
        )
    return data


# ------------------------------------------------------------------- векторы

def cosine(a: list, b: list) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def centroid(vectors: list) -> list:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def match(emb: list, reg: dict, threshold: float) -> list:
    """Возвращает [(имя, близость), ...] по убыванию близости."""
    scored = [
        (name, cosine(emb, entry["centroid"]))
        for name, entry in reg["speakers"].items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# Опорные точки, замеренные на pyannote community-1: ниже SAME_FLOOR лежат
# разные люди, выше SAME_CEIL — надёжно один и тот же. Проценты считаются
# по этому диапазону, а не по всему [-1..1], иначе разные голоса выглядят
# как 70%+ и число вводит в заблуждение.
DIFF_CEIL = 0.45
SAME_FLOOR = 0.95


def as_percent(sim: float) -> int:
    """Косинус → уверенность «тот же человек» в процентах."""
    frac = (sim - DIFF_CEIL) / (SAME_FLOOR - DIFF_CEIL)
    return max(0, min(100, round(frac * 100)))


# ------------------------------------------------------------------- команды

def cmd_enroll(args) -> None:
    data = load_result(Path(args.result))
    reg = load_registry()
    embeddings = data["speaker_embeddings"]

    pairs = []
    for item in args.map:
        if "=" not in item:
            sys.exit(f"Ожидается SPEAKER_XX=Имя, получено: {item}")
        label, name = item.split("=", 1)
        pairs.append((label.strip(), name.strip()))

    for label, name in pairs:
        if label not in embeddings:
            sys.exit(
                f"{label} нет в файле. Доступные: {', '.join(sorted(embeddings))}"
            )
        emb = embeddings[label]
        entry = reg["speakers"].setdefault(
            name, {"samples": [], "centroid": None, "sources": []}
        )
        entry["samples"].append(emb)
        entry["centroid"] = centroid(entry["samples"])
        entry["sources"].append(
            {
                "file": str(Path(args.result).resolve()),
                "label": label,
                "added": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        entry["dim"] = len(emb)
        print(f"  {label} → {name}  (образцов: {len(entry['samples'])})")

    save_registry(reg)
    print(f"\nРеестр: {REGISTRY}")


def print_runners_up(scored: list, top: int, start: int = 1) -> None:
    """Печатает остальных кандидатов после победителя, от самого близкого."""
    rest = scored[start:top]
    for i, (name, sim) in enumerate(rest, start=start + 1):
        print(f"       {i}. {name:<20} {as_percent(sim):>3}%  cos={sim:.3f}")


def cmd_identify(args) -> None:
    data = load_result(Path(args.result))
    reg = load_registry()
    if not reg["speakers"]:
        sys.exit("Реестр пуст — сначала spkreg enroll")

    for label, emb in sorted(data["speaker_embeddings"].items()):
        scored = match(emb, reg, args.threshold)
        best, sim = scored[0]
        if sim >= args.threshold:
            print(f"{label}  →  {best}   {as_percent(sim)}%  cos={sim:.3f}")
            print_runners_up(scored, args.top)
        else:
            print(f"{label}  →  неизвестный   (ближайший ниже порога {args.threshold})")
            print_runners_up(scored, args.top, start=0)


def resolve_names(data: dict, reg: dict, threshold: float) -> dict:
    """label → отображаемое имя."""
    names = {}
    for label, emb in data["speaker_embeddings"].items():
        if not reg["speakers"]:
            names[label] = label
            continue
        best, sim = match(emb, reg, threshold)[0]
        names[label] = f"{best}" if sim >= threshold else label
    return names


def cmd_apply(args) -> None:
    data = load_result(Path(args.result))
    reg = load_registry()
    names = resolve_names(data, reg, args.threshold)

    if args.format == "json":
        for seg in data.get("segments", []):
            if "speaker" in seg:
                seg["speaker_label"] = seg["speaker"]
                seg["speaker"] = names.get(seg["speaker"], seg["speaker"])
        out = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        lines = []
        for seg in data.get("segments", []):
            label = seg.get("speaker", "?")
            name = names.get(label, label)
            lines.append(
                f"[{fmt_ts(seg['start'])} → {fmt_ts(seg['end'])}] {name}: "
                f"{seg['text'].strip()}"
            )
        out = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Записано: {args.output}")
    else:
        print(out)


def fmt_ts(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def cmd_list(args) -> None:
    reg = load_registry()
    if not reg["speakers"]:
        print("Реестр пуст")
        return
    print(f"{REGISTRY}\n")
    for name, entry in sorted(reg["speakers"].items()):
        last = entry["sources"][-1]["added"] if entry["sources"] else "?"
        print(
            f"{name:<30} образцов: {len(entry['samples']):<3} "
            f"dim: {entry.get('dim', '?'):<5} обновлён: {last}"
        )
        if args.verbose:
            for src in entry["sources"]:
                print(f"    {src['label']:<12} {src['file']}")


def cmd_forget(args) -> None:
    if not args.name and not args.source:
        sys.exit("Укажи имя, --source или и то и другое")

    reg = load_registry()
    if args.name and args.name not in reg["speakers"]:
        sys.exit(f"Нет такого спикера: {args.name}")

    targets = [args.name] if args.name else list(reg["speakers"])

    # Целиком человека — только когда источник не задан.
    if args.name and not args.source:
        del reg["speakers"][args.name]
        save_registry(reg)
        print(f"Удалён целиком: {args.name}")
        return

    src = str(Path(args.source).resolve())
    removed = []
    for name in targets:
        entry = reg["speakers"][name]
        samples, sources = entry["samples"], entry["sources"]
        if len(samples) != len(sources):
            sys.exit(
                f"{name}: {len(samples)} образцов против {len(sources)} записей "
                "в истории — списки разъехались, точечное удаление небезопасно. "
                "Удаляй целиком по имени."
            )

        keep = [i for i, s in enumerate(sources) if s["file"] != src]
        if len(keep) == len(samples):
            continue

        dropped = len(samples) - len(keep)
        if not keep:
            del reg["speakers"][name]
            removed.append(f"{name}: убран последний образец, запись удалена")
            continue

        entry["samples"] = [samples[i] for i in keep]
        entry["sources"] = [sources[i] for i in keep]
        entry["centroid"] = centroid(entry["samples"])
        removed.append(f"{name}: -{dropped} образец(ов), осталось {len(keep)}")

    if not removed:
        print(f"Образцов из этого файла не найдено: {src}")
        return

    save_registry(reg)
    for line in removed:
        print(f"  {line}")


# ---------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(
        prog="spkreg", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="запомнить голоса из размеченного файла")
    e.add_argument("result", help="JSON от whispermlx")
    e.add_argument("map", nargs="+", metavar="SPEAKER_XX=Имя")
    e.set_defaults(func=cmd_enroll)

    i = sub.add_parser("identify", help="сопоставить спикеров с реестром")
    i.add_argument("result")
    i.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    i.add_argument(
        "--top",
        type=int,
        default=3,
        help="сколько кандидатов показывать (по умолчанию 3)",
    )
    i.set_defaults(func=cmd_identify)

    a = sub.add_parser("apply", help="подставить имена в транскрипт")
    a.add_argument("result")
    a.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    a.add_argument("-f", "--format", choices=["txt", "json"], default="txt")
    a.add_argument("-o", "--output")
    a.set_defaults(func=cmd_apply)

    l = sub.add_parser("list", help="показать известных спикеров")
    l.add_argument("-v", "--verbose", action="store_true")
    l.set_defaults(func=cmd_list)

    f = sub.add_parser(
        "forget",
        help="удалить спикера целиком или отдельные образцы по файлу-источнику",
    )
    f.add_argument("name", nargs="?", help="имя из реестра; без --source — удалить целиком")
    f.add_argument(
        "--source",
        help="JSON-источник: убрать образцы, пришедшие из него "
        "(с именем — только у этого человека, без имени — у всех)",
    )
    f.set_defaults(func=cmd_forget)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
