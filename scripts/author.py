#!/usr/bin/env python3
"""Helpers for writing path text by hand (workflow and conventions: docs/authoring.md).

    python scripts/author.py find slope 45 2x2            # search the part vocabulary
    python scripts/author.py gallery out.png 3001 --find "slope curved"   # contact sheet of parts
    python scripts/author.py part "brick 2x4"             # connector letters, part-local positions
    python scripts/author.py part "brick 2x4" --png out.png --sub stud,hole  # ... drawn on the part
    python scripts/author.py check model.txt              # parse, place, collide: one row per part
    python scripts/author.py check model.txt --conns b,d  # also these parts' connectors, in world space
    python scripts/author.py check model.txt -o model.ldr --path clean.txt
    python scripts/author.py try model.txt "animal horse complete | white" "bj stud hole {a,b} open a {90,270}"
    python scripts/author.py view model.txt out.png --labels  # fast WebGL preview grid (iso, front, right, top)
    python scripts/author.py view model.txt out.png --views 120/20 --target c --zoom 3 --highlight c --conns c
    python scripts/author.py render model.txt out/ --views 4  # slow, photoreal Blender render
    python scripts/author.py render model.txt out/ --turntable 8 --anim mp4,gif  # an 8 s turntable
    python scripts/author.py render model.txt out/ --turntable 12 --falling  # built brick by brick as it turns
    python scripts/author.py publish model.txt ../bricknet-models --image out/model_0000.png  # add to the gallery

Drafts may contain blank lines and '#' comments; they are stripped before parsing, and --path writes
the canonical path text. Node labels must match their position (a, b, ..., z, aa, ...): the parser
ignores them and resolves edge references positionally, so `check` rejects a mislabeled node.

Edge lines (the full grammar is in bricknet/tree.py):
    stud:  PARENT stud  PSUB PCONN CSUB CCONN YAW            PSUB/CSUB a mating pair, e.g. stud+hole
    axle:  PARENT axle  PSUB PCONN CSUB CCONN FLIP YAW SLIDE   e.g. pin+socket, bar+clip, axle+cross
    hinge: PARENT hinge SUB POL PCONN CCONN FLIP YAW         SUB/POL as listed by `part`; child is the
    ball:  PARENT ball  SUB POL PCONN CCONN RX RY RZ         opposite polarity
    fixed: PARENT fixed SUB POL PCONN CCONN
The child is placed so its connector frame meets the parent's, turned by YAW degrees about the mating
axis. Coordinates are LDraw LDU: 20 per stud pitch, 8 per plate, 24 per brick, and -y is up.
"""

import argparse
import itertools
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bricknet import collision
from bricknet.core import FixedEdge, Tree, _AXLE_MATES, _STUD_MATES
from bricknet.data import load_catalog, load_connector_list, load_connectors
from bricknet.graph import decode_graph, graph_to_ldr, parse_ldr, tree_to_graph
from bricknet.tree import ParseError, i2id, id2i, parse_edge, parse_node, serialize_tree

_DIRS = {
    (1, 0, 0): "+x",
    (-1, 0, 0): "-x",
    (0, 1, 0): "+y(down)",
    (0, -1, 0): "-y(up)",
    (0, 0, 1): "+z",
    (0, 0, -1): "-z",
}


def _mates(sub):
    pairs = _STUD_MATES | _AXLE_MATES
    return sorted({b if a == sub else a for a, b in pairs if sub in (a, b)})


def _family(kind, sub):
    if kind in ("stud", "hole"):
        return "stud"
    if kind == "axle":
        return "axle"
    return kind


def _num(v):
    return str(int(round(v))) if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def _vec(v):
    return "(" + ", ".join(f"{_num(x):>4}" for x in v) + ")"


def _dir(v):
    r = tuple(int(round(x)) for x in v)
    return _DIRS.get(r, "") if np.allclose(v, r, atol=0.02) else "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def _resolve(query):
    """Part id from an exact name, a stem, or a unique search hit."""
    cat = load_catalog()
    if query in cat.name_to_id:
        return cat.name_to_id[query]
    if query.lower().removesuffix(".dat") in cat.stem_to_id:
        return cat.stem_to_id[query.lower().removesuffix(".dat")]
    hits = _search(query.split())
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"{query!r} is not an exact name or stem; candidates:\n" + "\n".join(_find_rows(hits[:20])))


def _search(tokens):
    cat = load_catalog()
    tokens = [t.lower() for t in tokens]
    hits = [pid for pid, name in enumerate(cat.id_to_name) if all(t in name for t in tokens)]
    phrase = " ".join(tokens)
    return sorted(
        hits,
        key=lambda p: (
            cat.id_to_name[p] != phrase,
            not cat.id_to_name[p].startswith(phrase),
            len(cat.id_to_name[p]),
            cat.id_to_name[p],
        ),
    )


def _summary(pid):
    counts = {}
    for kind, sub, pol, _, _ in load_connector_list()[pid]:
        key = sub if pol is None else f"{_family(kind, sub)} {i2id(int(sub))}/{pol}"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def _find_rows(pids):
    cat = load_catalog()
    return [f"  {cat.id_to_stem[p]:>12}  {cat.id_to_name[p]:<50}  {_summary(p)}" for p in pids]


def _bounds_text(pid):
    try:
        lo, hi = collision._bounds(pid)
    except (FileNotFoundError, RuntimeError):
        return "bounds unknown (no collider)"
    return "bounds " + "  ".join(f"{a} {_num(lo[i])}..{_num(hi[i])}" for i, a in enumerate("xyz"))


_MARK_COLORS = {
    **dict.fromkeys(["stud", "open"], "#c62828"),
    **dict.fromkeys(["hole", "tube", "post"], "#1565c0"),
    **dict.fromkeys(["bar", "axle", "pin"], "#2e7d32"),
    **dict.fromkeys(["clip", "cross", "socket"], "#00838f"),
    "hinge": "#ef6c00",
    "ball": "#6a1b9a",
    "fixed": "#616161",
}


def _world_conns(pid, m):
    """Connectors of part pid placed at m, in canonical order: [(label, sub, pos, axis)], where label is the
    text an edge line uses, e.g. "hole c" or "hinge h on a"."""
    frames = load_connectors()[pid]
    out = []
    for kind, sub, pol, idx, _ in load_connector_list()[pid]:
        f = m @ frames[(sub, pol)][idx]
        group = sub if pol is None else f"{_family(kind, sub)} {i2id(int(sub))} {pol}"
        out.append((f"{group} {i2id(idx)}", sub if pol is None else _family(kind, sub), f[:3, 3], -f[:3, 1]))
    return out


def _markers(pid, m, mated=None, subs=None):
    """Viewer markers for a placed part's connectors; mated maps flat connector index -> other part index."""
    out = []
    for flat, (label, sub, pos, _) in enumerate(_world_conns(pid, m)):
        if subs and sub not in subs and label.split(" ")[0] not in subs:
            continue
        other = (mated or {}).get(flat)
        out.append(
            {
                "pos": [float(v) for v in pos],
                "label": label if other is None else f"{label}*{i2id(other)}",
                "color": _MARK_COLORS.get(sub, "#333333"),
                "dim": other is not None,
            }
        )
    return out


def _mated(ldr, n):
    """From the realized LDR: ({(part, flat conn): other part}, linked-neighbor count per part)."""
    links = parse_ldr(ldr).edges
    used, degree = {}, [0] * n
    for e in links:
        a, b = int(e["a"]), int(e["b"])
        used[(a, int(e["a_conn"]))], used[(b, int(e["b_conn"]))] = b, a
    for a, b in {(min(int(e["a"]), int(e["b"])), max(int(e["a"]), int(e["b"]))) for e in links}:
        degree[a] += 1
        degree[b] += 1
    return used, degree


def cmd_find(args):
    hits = _search(args.query)
    print("\n".join(_find_rows(hits[: args.limit])))
    if len(hits) > args.limit:
        print(f"  ... {len(hits) - args.limit} more (--limit)")
    if not hits:
        sys.exit("no matches")


def cmd_part(args):
    cat = load_catalog()
    pid = _resolve(" ".join(args.query))
    if args.png:
        from viewer import driver  # scripts/viewer

        part = {
            "id": "a",
            "stem": cat.id_to_stem[pid],
            "color": cat.color_to_code[args.color],
            "matrix": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        }
        opts = _shoot_opts(args)
        opts["markers"] = _markers(pid, np.eye(4), subs=set(_ids(args.sub)))
        with driver.Session([part]) as session:
            session.shoot(Path(args.png), **opts)
        print(args.png)
    print(f"{cat.id_to_name[pid]}  [{cat.id_to_stem[pid]}]  {_bounds_text(pid)}  (LDU, -y up)")
    frames = load_connectors()[pid]
    conns = load_connector_list()[pid]
    for (sub, pol), group in frames.items():
        kind = next(c[0] for c in conns if c[1] == sub and c[2] == pol)
        fam = _family(kind, sub)
        if pol is None:
            print(f"  {sub}  ({fam} edge; mates {'/'.join(_mates(sub))})")
        else:
            other = "on" if pol == "in" else "in"
            print(f"  {fam} {i2id(int(sub))} {pol}  (subtype {sub}; the other side is {i2id(int(sub))} {other})")
        lengths = [c[4] for c in conns if c[1] == sub and c[2] == pol]
        for k, f in enumerate(group):
            extra = f"  len {_num(lengths[k])}" if lengths[k] else ""
            print(f"    {i2id(k):>3} {_vec(f[:3, 3])}  axis {_dir(-f[:3, 1])}{extra}")


def _strip(text):
    """(line_no, line) pairs with blank and '#' lines removed."""
    return [(i, ln.rstrip()) for i, ln in enumerate(text.split("\n"), 1) if ln.strip() and not ln.startswith("#")]


def _parse(lines):
    """Incremental parse with line-accurate errors. Returns (tree, error message or None)."""
    cat = load_catalog()
    parts, edges = [], []
    i = 0
    no, cur, hint = None, None, ""
    try:
        while i < len(lines):
            no, cur = lines[i]
            want = i2id(len(parts))
            label = cur.split(" ", 1)[0]
            if label != want:
                raise ParseError(f"node label {label!r} should be {want!r} (labels are positional)")
            part = parse_node(cur, cat)
            if parts:
                if i + 1 >= len(lines):
                    raise ParseError("node has no edge line after it")
                no, cur = lines[i + 1]
                hint = f"child {want} = {cat.id_to_name[part.part_id]}"
                try:
                    parent = id2i(cur.split(" ", 1)[0])
                    hint = f"parent {i2id(parent)} = {cat.id_to_name[parts[parent].part_id]}, {hint}"
                except (ParseError, IndexError):
                    pass
                edges.append(parse_edge(cur, parts, part.part_id, cat))
                hint = ""
                i += 1
            parts.append(part)
            i += 1
    except ParseError as e:
        return Tree(tuple(parts), tuple(edges)), f"line {no}: {e}\n    {cur}" + (f"\n    ({hint})" if hint else "")
    return Tree(tuple(parts), tuple(edges)), None


def _orientation(m):
    r = m[:3, :3]
    if abs(r[1, 1] - 1) < 1e-6:
        return f"yaw {round(math.degrees(math.atan2(r[0, 2], r[0, 0]))) % 360}"
    if abs(r[1, 1] + 1) < 1e-6:
        return f"upside-down, yaw {round(math.degrees(math.atan2(r[0, 2], r[0, 0]))) % 360}"
    return f"top faces {_dir(r @ np.array([0.0, -1.0, 0.0]))}"


def _collisions(tree, mats):
    """[(i, j, depth)] for every overlapping pair, i placed after j; fixed children ignore their parent."""
    fixed = {i + 1: e.parent for i, e in enumerate(tree.edges) if isinstance(e, FixedEdge)}
    pids = [p.part_id for p in tree.parts]
    out = []
    with collision.CollisionScene() as scene:
        for i, (pid, m) in enumerate(zip(pids, mats)):
            for j in scene.check(pid, m, exclude=fixed.get(i, -1)):
                out.append((i, j, collision.penetration(pid, m, pids[j], mats[j])))
            scene.add(pid, m)
    return out


def cmd_check(args):
    cat = load_catalog()
    text = Path(args.file).read_text() if args.file != "-" else sys.stdin.read()
    tree, err = _parse(_strip(text))
    if err:
        print(f"PARSE ERROR {err}")
        print(f"(parts placed before the error: {len(tree.parts)})\n")
    if not tree.parts:
        sys.exit(1)

    graph = tree_to_graph(tree)
    mats = [np.asarray(m) for m in decode_graph(graph)]
    pids = [p.part_id for p in tree.parts]
    hits = _collisions(tree, mats)
    hit_of = {}
    for i, j, d in hits:
        hit_of.setdefault(i, []).append(f"{i2id(j)}({d:.1f})")

    # every mated connector pair in the realized model, tree edges or not
    ldr = graph_to_ldr(graph)
    used, degree = _mated(ldr, len(pids))

    print(
        "id   part                                      color                 origin              "
        "orientation   links  collides with (depth LDU)"
    )
    lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
    for i, (p, m) in enumerate(zip(tree.parts, mats)):
        try:
            lo, hi = collision._world_aabb(*collision._bounds(p.part_id), m)
            lo_all, hi_all = np.minimum(lo_all, lo), np.maximum(hi_all, hi)
        except (FileNotFoundError, RuntimeError):
            pass
        if args.last and i < len(pids) - args.last and i not in hit_of:
            continue
        name, color = cat.id_to_name[p.part_id], cat.code_to_color[p.color]
        print(
            f"{i2id(i):<4} {name[:41]:<41} {color[:21]:<21} {_vec(m[:3, 3]):<19} {_orientation(m):<13} "
            f"{degree[i]:>5}  {' '.join(hit_of.get(i, []))}"
        )

    if np.isfinite(lo_all).all():
        size = hi_all - lo_all
        print(
            f"\nextent x {_num(lo_all[0])}..{_num(hi_all[0])}  y {_num(lo_all[1])}..{_num(hi_all[1])}  "
            f"z {_num(lo_all[2])}..{_num(hi_all[2])}  ~ {size[0] / 20:.1f} x {size[2] / 20:.1f} studs, "
            f"{size[1] / 8:.1f} plates tall"
        )
    print(f"{len(pids)} parts, {len(hits)} colliding pairs" + (", PARSE ERROR" if err else ""))

    for label in [s for arg in args.conns for s in arg.split(",") if s]:
        k = id2i(label)
        if k >= len(pids):
            sys.exit(f"--conns {label}: no such part")
        print(f"\n{label} {cat.id_to_name[pids[k]]}: connectors in world space (* = mated, with part)")
        for flat, (name, _, pos, axis) in enumerate(_world_conns(pids[k], mats[k])):
            group, letter = name.rsplit(" ", 1)
            mark = f"* {i2id(used[(k, flat)])}" if (k, flat) in used else ""
            print(f"  {group:<14} {letter:>3} {_vec(pos)}  axis {_dir(axis):<10} {mark}")

    if args.output:
        Path(args.output).write_text(ldr)
    if args.path:
        Path(args.path).write_text(serialize_tree(tree))
    if err or (hits and not args.allow_collisions):
        sys.exit(1)


def _realize(src):
    """(LDR text, part ids, world matrices) of a .ldr file or a path-text draft (exits on a parse error)."""
    if src.suffix == ".ldr":
        ldr = src.read_text()
        graph = parse_ldr(ldr)
        return ldr, list(graph.part_ids), list(graph.transforms)
    tree, err = _parse(_strip(src.read_text()))
    if err:
        sys.exit(f"PARSE ERROR {err}")
    graph = tree_to_graph(tree)
    return graph_to_ldr(graph), [p.part_id for p in tree.parts], [np.asarray(m) for m in decode_graph(graph)]


def _ldr_text(src):
    """LDR text of a .ldr file, or of a path-text draft (exits on a parse error)."""
    return src.read_text() if src.suffix == ".ldr" else _realize(src)[0]


# bricknet-render's CLI has no turntable flags, though its RenderOptions takes TurntableOptions. This runs in its
# own process, as the CLI does: it parses the CLI's flags exactly as bricknet-render would, adds the turntable, and
# renders. RenderOptions derives settings from the raw values once (lights for a floor, say), so the values are
# taken from parse_args before it builds them.
_TURNTABLE = """
import json, sys
from bricknet_render import render as r

values = {}
options, r.RenderOptions = r.RenderOptions, lambda **kw: values.update(kw)
r.parse_args(sys.argv[2:])
r.RenderOptions = options
args = options(**values, turntable=r.TurntableOptions(**json.loads(sys.argv[1])))
r.build_scene(args)
r.configure_output(args)
r.save_blend(args)
if not args.no_render:
    r.render_turntable(args)
"""


def _build_steps(text, per=12):
    """LDR text with `0 STEP` lines added so a build animation reveals the model from the bottom up: parts sorted by
    the height of their lowest point, then swept across each layer, about `per` parts to a step. Text that already
    has steps is returned unchanged."""
    if re.search(r"^0 (ROT)?STEP\b", text, re.M):
        return text
    cat = load_catalog()
    head, parts = [], []
    for line in text.splitlines(keepends=True):
        f = line.split()
        if f and f[0] == "1":
            t = np.array(f[2:5], float)
            R = np.array(f[5:14], float).reshape(3, 3)
            lo, hi = collision._bounds(cat.stem_to_id[f[14].removesuffix(".dat")])
            w = np.array(list(itertools.product(*zip(lo, hi)))) @ R.T + t
            key = (-round(w[:, 1].max() / 8), round(w[:, 0].mean() / 20), round(w[:, 2].mean() / 20))
            parts.append((key, line))  # -y is up: the lowest point has the largest y
        elif not parts:
            head.append(line)
    parts.sort(key=lambda p: p[0])
    out = list(head)
    for k, (key, line) in enumerate(parts):
        if k and (k % per == 0 or key[0] != parts[k - 1][0][0]):
            out.append("0 STEP\n")
        out.append(line)
    return "".join(out)


def _encode(frames_dir, out, stem, fps, kinds, width):
    """Encode the turntable's PNG frames as an H.264 MP4 and/or a GIF (at most 12 fps and 480 px wide) in `out`."""
    frames = ["-framerate", f"{fps:g}", "-i", str(frames_dir / f"{stem}_%04d.png")]
    ffmpeg = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y"]
    made = []
    if "mp4" in kinds:
        made.append(out / f"{stem}.mp4")
        even = "crop=trunc(iw/2)*2:trunc(ih/2)*2"  # yuv420p needs even dimensions
        enc = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        subprocess.run(ffmpeg + frames + ["-vf", even] + enc + [str(made[-1])], check=True)
    if "gif" in kinds:
        made.append(out / f"{stem}.gif")
        gif = (
            f"fps={min(fps, 12):g},scale={min(width, 480)}:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=stats_mode=diff:max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
        )
        subprocess.run(ffmpeg + frames + ["-vf", gif, str(made[-1])], check=True)
    return made


def cmd_render(args):
    src = Path(args.file)
    out = Path(args.out_dir)
    kinds = set(args.anim.split(",")) if args.turntable else set()
    if kinds - {"mp4", "gif", "frames"}:
        sys.exit(f"--anim takes mp4, gif and frames; got {args.anim}")
    if args.turntable and args.views != 1:
        sys.exit("--turntable and --views are mutually exclusive")
    if kinds - {"frames"} and not shutil.which("ffmpeg"):
        sys.exit("--anim mp4/gif needs ffmpeg on PATH (or use --anim frames)")
    args.build = args.build or args.falling
    if args.build and not args.turntable:
        sys.exit("--build and --falling animate a --turntable")
    out.mkdir(parents=True, exist_ok=True)
    if args.build:  # a copy with build steps; its name tells the animation from a plain turntable's
        ldr_path = out / (src.stem + "_build.ldr")
        ldr_path.write_text(_build_steps(_ldr_text(src)))
    elif src.suffix == ".ldr":
        ldr_path = src
    else:
        ldr_path = out / (src.stem + ".ldr")
        ldr_path.write_text(_ldr_text(src))
    stem = ldr_path.stem
    frames_dir = out / f"{stem}_frames"  # a turntable's frames would share the stills' names (model_0000.png)
    if args.turntable:
        shutil.rmtree(frames_dir, ignore_errors=True)  # stale frames past the new last one would be encoded too
    flags = [
        str(ldr_path),
        str(frames_dir if args.turntable else out),
        "--device",
        args.device,
        "--samples",
        str(args.samples),
        "--resolution",
        args.resolution,
    ]
    if args.turntable:
        spec = {
            "format": "png8",
            "fps": args.fps,
            "duration": args.turntable,
            "rotation_period": args.rotation or args.turntable,
            "clockwise": not args.counterclockwise,
            "build_steps": args.build,
            "falling": args.falling,
            "build_duration_ratio": args.build_share,
        }
        if not {"--background-mode", "--transparent"} & set(args.extra):
            flags += ["--background-mode", "floor"]  # frames on the white floor, as the stills are
        cmd = [sys.executable, "-c", _TURNTABLE, json.dumps(spec)] + flags
    else:
        cmd = [sys.executable, "-m", "bricknet_render"] + flags + ["--views", str(args.views)]
    with tempfile.TemporaryFile("w+") as log:
        rc = subprocess.call(cmd + args.extra, stdout=log, stderr=subprocess.STDOUT)
        if rc:
            log.seek(0)
            sys.exit(log.read()[-3000:] + f"\nbricknet-render exited {rc}")
    if not args.turntable:
        for png in sorted(out.glob("*.png")):
            print(png)
        return
    width = int(args.resolution.split("x")[0])
    made = _encode(frames_dir, out, stem, args.fps, kinds, width) if kinds - {"frames"} else []
    if "frames" in kinds:
        print(f"{frames_dir}/ ({len(list(frames_dir.glob('*.png')))} frames)")
    else:
        shutil.rmtree(frames_dir)
    for path in made:
        print(path)


def _ids(arg):
    return [t for t in (arg or "").split(",") if t]


def cmd_view(args):
    from viewer import driver  # scripts/viewer

    src = Path(args.file)
    ldr, pids, mats = _realize(src)
    parts = driver.ldr_parts(ldr)
    if args.html:
        missing = driver.export_html(parts, Path(args.html), title=src.stem)
        print(args.html + (f"  (no mesh for: {' '.join(missing)})" if missing else ""))
    if not args.out:
        return
    upto = None
    if args.upto:
        upto = int(args.upto) if args.upto.isdigit() else id2i(args.upto) + 1
    opts = dict(
        _shoot_opts(args),
        labels=args.labels,
        hide=_ids(args.hide),
        highlight=_ids(args.highlight),
        upto=upto,
        markers=[],
    )
    if args.conns:
        used, _ = _mated(ldr, len(pids))
        for label in _ids(args.conns):
            k = id2i(label)
            if k >= len(pids):
                sys.exit(f"--conns {label}: no such part")
            mated = {flat: other for (part, flat), other in used.items() if part == k}
            opts["markers"] += _markers(pids[k], mats[k], mated, set(_ids(args.sub)))
    with driver.Session(parts) as session:
        if session.missing:
            print(f"warning: no mesh for {' '.join(session.missing)}; those parts are not drawn", file=sys.stderr)
        views = session.shoot(Path(args.out), **opts)
    print(args.out)
    if args.labels:
        seen = set().union(*(v["visible"] for v in views))
        shown = [p["id"] for p in parts[:upto] if p["id"] not in _ids(args.hide)]
        unseen = [i for i in shown if i not in seen and (not args.highlight or i in _ids(args.highlight))]
        if unseen:
            print(f"not visible in any view: {' '.join(unseen)}")


def _shoot_opts(args):
    """Camera and layout options shared by the commands that render through the WebGL viewer."""
    target = args.target
    if target and "," in target:
        target = [float(v) for v in target.split(",")]
    return dict(
        views=args.views.split(","),
        cols=args.cols,
        size=args.size,
        zoom=args.zoom,
        pan=[float(v) for v in args.pan.split(",")],
        target=target,
        ortho=args.ortho,
        grid=not args.no_grid,
        edges=not args.no_edges,
    )


def cmd_gallery(args):
    from viewer import driver  # scripts/viewer

    cat = load_catalog()
    pids = [_resolve(q) for q in args.parts]
    if args.find:
        pids += _search(args.find.split())[: args.limit]
    if not pids:
        sys.exit("nothing to show: name parts or pass --find")
    color = cat.color_to_code[args.color]
    identity = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    parts = [{"id": i2id(k), "stem": cat.id_to_stem[p], "color": color, "matrix": identity} for k, p in enumerate(pids)]
    tiles = [
        {"view": args.view, "show": [part["id"]], "caption": f"{cat.id_to_stem[p]} {cat.id_to_name[p]}"}
        for part, p in zip(parts, pids)
    ]
    with driver.Session(parts) as session:
        session.shoot(Path(args.out), tiles=tiles, cols=args.cols, size=args.size, grid=False, ortho=args.ortho)
    print(args.out)


_GALLERY_INTRO = """# BrickNet models

Hand-authored LEGO models in [BrickNet](https://github.com/Synook/BrickNet) path text. Each folder holds
`model.txt` (the commented path text, the source of truth) and `preview.jpg`. Build `.ldr` files and
renders from a model with BrickNet's `scripts/author.py check -o` and `render`.

This index is generated by `scripts/author.py publish` / `index`; edit the models, not the table.

| preview | model | prompt | parts |
| --- | --- | --- | --- |
"""


def _model_meta(path):
    """(title, prompt, part count) from a model file: the first comment line is the title, a
    '# prompt: ...' comment line the prompt."""
    text = path.read_text()
    comments = [ln[1:].strip() for ln in text.split("\n") if ln.startswith("#")]
    prompts = [c.split(":", 1)[1].strip() for c in comments if c.lower().startswith("prompt:")]
    title = next((c for c in comments if not c.lower().startswith("prompt:")), path.parent.name)
    return title, prompts[0] if prompts else "", (len(_strip(text)) + 1) // 2


def _write_index(gallery):
    rows = []
    for model in sorted(gallery.glob("*/model.txt")):
        title, prompt, n = _model_meta(model)
        name = model.parent.name
        img = f'<img src="{name}/preview.jpg" width="240">' if (model.parent / "preview.jpg").exists() else ""
        cells = [img, f"[{title}]({name}/model.txt)", prompt, str(n)]
        rows.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |\n")
    (gallery / "README.md").write_text(_GALLERY_INTRO + "".join(rows))
    return len(rows)


def cmd_publish(args):
    from viewer import driver  # scripts/viewer

    src, gallery = Path(args.draft), Path(args.gallery)
    tree, err = _parse(_strip(src.read_text()))
    if err:
        sys.exit(f"PARSE ERROR {err}")
    graph = tree_to_graph(tree)
    mats = [np.asarray(m) for m in decode_graph(graph)]
    hits = _collisions(tree, mats)
    if hits and not args.force:
        sys.exit(f"{len(hits)} colliding pairs; fix them (see check) or pass --force")
    slug = args.name or (src.parent.name if src.name == "model.txt" else src.stem)
    dest = gallery / slug
    dest.mkdir(parents=True, exist_ok=True)
    if src.resolve() != (dest / "model.txt").resolve():
        (dest / "model.txt").write_text(src.read_text())
    preview = dest / "preview.jpg"
    if args.image:
        from PIL import Image

        im = Image.open(args.image).convert("RGB")
        im.thumbnail((args.width, args.width))
        im.save(preview, quality=88)
    else:
        with driver.Session(driver.ldr_parts(graph_to_ldr(graph))) as session:
            session.shoot(preview, views=[args.view], size=args.width, format="jpeg")
    n = _write_index(gallery)
    print(
        f"{dest / 'model.txt'}\n{preview} ({preview.stat().st_size // 1024} KB)\n{gallery / 'README.md'} ({n} models)"
    )


def cmd_index(args):
    print(f"{Path(args.gallery) / 'README.md'} ({_write_index(Path(args.gallery))} models)")


def _expand(text):
    """Every expansion of {x,y,...} alternatives in text, in order."""
    groups = re.findall(r"\{([^{}]*)\}", text)
    template = re.sub(r"\{[^{}]*\}", "{}", text)
    return [template.format(*combo) for combo in itertools.product(*(g.split(",") for g in groups))]


def cmd_try(args):
    lines = _strip(Path(args.file).read_text())
    base, err = _parse(lines)
    if err:
        sys.exit(f"PARSE ERROR in the draft {err}")
    want = i2id(len(base.parts))
    base_mats = [np.asarray(m) for m in decode_graph(tree_to_graph(base))] if base.parts else []
    variants = []
    for node in _expand(args.node):
        if node.split(" ", 1)[0] != want:
            node = f"{want} {node}"
        variants += [(node, edge) for edge in _expand(args.edge)]

    with collision.CollisionScene() as scene:
        for p, m in zip(base.parts, base_mats):
            scene.add(p.part_id, m)
        for k, (node, edge) in enumerate(variants, 1):
            tree, err = _parse(lines + [(0, node), (0, edge)])
            label = f"{k:>2}. {edge}" + ("" if len(set(n for n, _ in variants)) == 1 else f"   [{node}]")
            if err:
                first, *rest = err.split("\n")
                hint = rest[-1].strip() if len(rest) > 1 else ""
                print(f"{label}\n      PARSE ERROR {first.split(': ', 1)[-1]} {hint}")
                continue
            graph = tree_to_graph(tree)
            m = np.asarray(decode_graph(graph)[-1])
            pid = tree.parts[-1].part_id
            exclude = tree.edges[-1].parent if isinstance(tree.edges[-1], FixedEdge) else -1
            hits = [
                (j, collision.penetration(pid, m, base.parts[j].part_id, base_mats[j]))
                for j in scene.check(pid, m, exclude=exclude)
            ]
            used, _ = _mated(graph_to_ldr(graph), len(tree.parts))
            links = sorted({i2id(o) for (part, _), o in used.items() if part == len(tree.parts) - 1})
            clash = "  collides " + " ".join(f"{i2id(j)}({d:.1f})" for j, d in hits) if hits else ""
            print(f"{label}\n      {_vec(m[:3, 3])}  {_orientation(m):<13}  links {','.join(links) or '-'}{clash}")
    if args.append:
        node, edge = variants[args.append - 1]
        with open(args.file, "a") as f:
            f.write(f"{node}\n{edge}\n")
        print(f"appended variant {args.append} to {args.file}")


def _add_camera_args(p, views="iso,front,right,top"):
    p.add_argument(
        "--views",
        default=views,
        help="comma list of front, back, left, right, top, bottom, iso, iso-back, or AZ/EL degrees "
        "(azimuth 0 = front, 90 = right side; elevation 90 = straight down)",
    )
    p.add_argument("--cols", type=int, help="tiles per row (default: a square-ish grid)")
    p.add_argument("--size", type=int, default=512, help="tile size in pixels")
    p.add_argument("--zoom", type=float, default=1.0, help=">1 zooms in from the fitted view")
    p.add_argument("--target", help="center on a part id (e.g. c) or an LDU point x,y,z")
    p.add_argument("--pan", default="0,0", help="shift the view by X,Y frame widths, e.g. 0.25,0")
    p.add_argument("--ortho", action="store_true", help="orthographic projection")
    p.add_argument("--no-grid", action="store_true", help="omit the 1-stud ground grid")
    p.add_argument("--no-edges", action="store_true", help="omit part edge lines")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("find", help="search part names (every word must appear)")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_find)

    p = sub.add_parser("part", help="connector letters of one part, with part-local positions")
    p.add_argument("query", nargs="+", help="exact name, stem, or words matching one part")
    p.add_argument("--png", help="also render the part with every connector labelled, to this PNG")
    p.add_argument("--sub", help="label only these subtypes or families, e.g. stud,hole or hinge")
    p.add_argument("--color", default="light bluish grey", help="color name for --png")
    _add_camera_args(p)
    p.set_defaults(func=cmd_part)

    p = sub.add_parser("gallery", help="contact sheet: one tile per part, from names/stems and/or a search")
    p.add_argument("out", help="output PNG")
    p.add_argument("parts", nargs="*", help="exact names or stems (quote names with spaces)")
    p.add_argument("--find", help="also add the top search hits for these words")
    p.add_argument("--limit", type=int, default=15, help="search hits to add with --find")
    p.add_argument("--color", default="light bluish grey", help="color name")
    p.add_argument("--view", default="iso", help="one view, as for view --views")
    p.add_argument("--ortho", action="store_true")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--size", type=int, default=240)
    p.set_defaults(func=cmd_gallery)

    p = sub.add_parser("publish", help="check a draft, add it to a gallery checkout with a preview, update its index")
    p.add_argument("draft", help="draft path text (may already be GALLERY/NAME/model.txt)")
    p.add_argument("gallery", help="gallery checkout, e.g. ../bricknet-models")
    p.add_argument("--name", help="folder name (default: the draft's stem, or its folder for model.txt)")
    p.add_argument("--image", help="preview from this image (e.g. a render), downscaled; default a WebGL view")
    p.add_argument("--view", default="35/25", help="WebGL preview view (default 35/25)")
    p.add_argument("--width", type=int, default=560, help="preview size in pixels (longest side)")
    p.add_argument("--force", action="store_true", help="publish despite collisions")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("index", help="regenerate a gallery checkout's README index")
    p.add_argument("gallery")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("try", help="place each variant of the next part/edge; report placement, links, collisions")
    p.add_argument("file", help="draft path text; the variants follow its last part")
    p.add_argument("node", help="node line, id optional, e.g. 'animal horse complete | white'")
    p.add_argument("edge", help="edge line; {x,y} alternatives expand, e.g. 'bj stud hole {a,b} open a {90,270}'")
    p.add_argument("--append", type=int, metavar="K", help="append variant K to the draft")
    p.set_defaults(func=cmd_try)

    p = sub.add_parser("check", help="parse, place, and collision-check a draft")
    p.add_argument("file", help="draft path text, or - for stdin")
    p.add_argument("--conns", action="append", default=[], help="part ids whose world connectors to list")
    p.add_argument("-o", "--output", help="write the realized .ldr here")
    p.add_argument("--path", help="write canonical path text (comments stripped) here")
    p.add_argument("--allow-collisions", action="store_true", help="exit 0 despite collisions")
    p.add_argument("--last", type=int, help="table rows only for the last N parts (and any colliding part)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser(
        "render", help="render a draft or .ldr with bricknet-render on the CPU; other flags pass through to it"
    )
    p.add_argument("file", help="draft path text or .ldr")
    p.add_argument("out_dir")
    p.add_argument("--views", type=int, default=1)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--resolution", default="512x512")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    p.add_argument("--turntable", type=float, metavar="SECONDS", help="render a turntable animation this long")
    p.add_argument("--fps", type=float, default=24, help="turntable frame rate (default 24)")
    p.add_argument("--rotation", type=float, metavar="SECONDS", help="seconds per turn (default: the whole animation)")
    p.add_argument("--counterclockwise", action="store_true", help="turn the other way")
    p.add_argument(
        "--build",
        action="store_true",
        help="reveal the model step by step as it turns (bottom up, unless the .ldr has its own 0 STEP lines)",
    )
    p.add_argument("--falling", action="store_true", help="--build, with each step dropping into place")
    p.add_argument("--build-share", type=float, default=0.8, help="share of the animation spent building (0.8)")
    p.add_argument("--anim", default="mp4", help="comma list of mp4, gif, frames (keep the PNGs); default mp4")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("view", help="fast WebGL preview: a grid of camera views in one PNG")
    p.add_argument("file", help="draft path text or .ldr")
    p.add_argument("out", nargs="?", help="output PNG")
    _add_camera_args(p)
    p.add_argument("--labels", action="store_true", help="label parts with their path-text ids")
    p.add_argument("--highlight", help="comma list of part ids to emphasize; the rest are ghosted")
    p.add_argument("--hide", help="comma list of part ids to leave out")
    p.add_argument("--upto", help="show only the first N parts, or parts through id X")
    p.add_argument("--conns", help="comma list of part ids whose connectors to mark (grey = mated, *other part)")
    p.add_argument("--sub", help="with --conns: mark only these subtypes or families, e.g. stud,hole")
    p.add_argument("--html", help="also write a standalone interactive viewer page")
    p.set_defaults(func=cmd_view)

    args, unknown = ap.parse_known_args()
    if args.command == "render":
        args.extra = [a for a in unknown if a != "--"]  # e.g. --yaw0 35 --pitch -20
    elif unknown:
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.func(args)


if __name__ == "__main__":
    main()
