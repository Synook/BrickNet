#!/usr/bin/env python3
"""Helpers for writing path text by hand.

    python scripts/author.py find slope 45 2x2            # search the part vocabulary
    python scripts/author.py part "brick 2x4"             # connector letters, part-local positions
    python scripts/author.py check model.txt              # parse, place, collide: one row per part
    python scripts/author.py check model.txt --conns b,d  # also these parts' connectors, in world space
    python scripts/author.py check model.txt -o model.ldr --path clean.txt
    python scripts/author.py view model.txt out.png --labels  # fast WebGL preview grid (iso, front, right, top)
    python scripts/author.py view model.txt out.png --views 120/20 --target c --zoom 3 --highlight c
    python scripts/author.py render model.txt out/ --views 4  # slow, photoreal Blender render

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
import math
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
    links = parse_ldr(ldr).edges
    used = {}  # (part, flat conn) -> other part
    degree = [0] * len(pids)
    for e in links:
        a, b = int(e["a"]), int(e["b"])
        used[(a, int(e["a_conn"]))], used[(b, int(e["b_conn"]))] = b, a
    for a, b in {(min(int(e["a"]), int(e["b"])), max(int(e["a"]), int(e["b"]))) for e in links}:
        degree[a] += 1
        degree[b] += 1

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
        m, conns = mats[k], load_connector_list()[pids[k]]
        print(f"\n{label} {cat.id_to_name[pids[k]]}: connectors in world space (* = mated, with part)")
        frames = load_connectors()[pids[k]]
        for flat, (kind, sub, pol, idx, _) in enumerate(conns):
            f = m @ frames[(sub, pol)][idx]
            group = sub if pol is None else f"{_family(kind, sub)} {i2id(int(sub))} {pol}"
            mark = f"* {i2id(used[(k, flat)])}" if (k, flat) in used else ""
            print(f"  {group:<14} {i2id(idx):>3} {_vec(f[:3, 3])}  axis {_dir(-f[:3, 1]):<10} {mark}")

    if args.output:
        Path(args.output).write_text(ldr)
    if args.path:
        Path(args.path).write_text(serialize_tree(tree))
    if err or (hits and not args.allow_collisions):
        sys.exit(1)


def _ldr_text(src):
    """LDR text of a .ldr file, or of a path-text draft (exits on a parse error)."""
    if src.suffix == ".ldr":
        return src.read_text()
    tree, err = _parse(_strip(src.read_text()))
    if err:
        sys.exit(f"PARSE ERROR {err}")
    return graph_to_ldr(tree_to_graph(tree))


def cmd_render(args):
    src = Path(args.file)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".ldr":
        ldr_path = src
    else:
        ldr_path = out / (src.stem + ".ldr")
        ldr_path.write_text(_ldr_text(src))
    cmd = [
        sys.executable,
        "-m",
        "bricknet_render",
        str(ldr_path),
        str(out),
        "--device",
        args.device,
        "--samples",
        str(args.samples),
        "--resolution",
        args.resolution,
        "--views",
        str(args.views),
    ]
    with tempfile.TemporaryFile("w+") as log:
        rc = subprocess.call(cmd + args.extra, stdout=log, stderr=subprocess.STDOUT)
        if rc:
            log.seek(0)
            sys.exit(log.read()[-3000:] + f"\nbricknet-render exited {rc}")
    for png in sorted(out.glob("*.png")):
        print(png)


def _ids(arg):
    return [t for t in (arg or "").split(",") if t]


def cmd_view(args):
    from viewer import driver  # scripts/viewer

    src = Path(args.file)
    parts = driver.ldr_parts(_ldr_text(src))
    if args.html:
        missing = driver.export_html(parts, Path(args.html), title=src.stem)
        print(args.html + (f"  (no mesh for: {' '.join(missing)})" if missing else ""))
    if not args.out:
        return
    upto = None
    if args.upto:
        upto = int(args.upto) if args.upto.isdigit() else id2i(args.upto) + 1
    target = args.target
    if target and "," in target:
        target = [float(v) for v in target.split(",")]
    opts = dict(
        views=args.views.split(","),
        cols=args.cols,
        size=args.size,
        zoom=args.zoom,
        pan=[float(v) for v in args.pan.split(",")],
        target=target,
        ortho=args.ortho,
        labels=args.labels,
        grid=not args.no_grid,
        edges=not args.no_edges,
        hide=_ids(args.hide),
        highlight=_ids(args.highlight),
        upto=upto,
    )
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("find", help="search part names (every word must appear)")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_find)

    p = sub.add_parser("part", help="connector letters of one part, with part-local positions")
    p.add_argument("query", nargs="+", help="exact name, stem, or words matching one part")
    p.set_defaults(func=cmd_part)

    p = sub.add_parser("check", help="parse, place, and collision-check a draft")
    p.add_argument("file", help="draft path text, or - for stdin")
    p.add_argument("--conns", action="append", default=[], help="part ids whose world connectors to list")
    p.add_argument("-o", "--output", help="write the realized .ldr here")
    p.add_argument("--path", help="write canonical path text (comments stripped) here")
    p.add_argument("--allow-collisions", action="store_true", help="exit 0 despite collisions")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("render", help="render a draft or .ldr with bricknet-render on the CPU")
    p.add_argument("file", help="draft path text or .ldr")
    p.add_argument("out_dir")
    p.add_argument("--views", type=int, default=1)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--resolution", default="512x512")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    p.add_argument("extra", nargs=argparse.REMAINDER, help="further bricknet-render flags, after --")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("view", help="fast WebGL preview: a grid of camera views in one PNG")
    p.add_argument("file", help="draft path text or .ldr")
    p.add_argument("out", nargs="?", help="output PNG")
    p.add_argument(
        "--views",
        default="iso,front,right,top",
        help="comma list of front, back, left, right, top, bottom, iso, iso-back, or AZ/EL degrees "
        "(azimuth 0 = front, 90 = right side; elevation 90 = straight down)",
    )
    p.add_argument("--cols", type=int, help="tiles per row (default: a square-ish grid)")
    p.add_argument("--size", type=int, default=512, help="tile size in pixels")
    p.add_argument("--zoom", type=float, default=1.0, help=">1 zooms in from the fitted view")
    p.add_argument("--target", help="center on a part id (e.g. c) or an LDU point x,y,z")
    p.add_argument("--pan", default="0,0", help="shift the view by X,Y frame widths, e.g. 0.25,0")
    p.add_argument("--ortho", action="store_true", help="orthographic projection")
    p.add_argument("--labels", action="store_true", help="label parts with their path-text ids")
    p.add_argument("--highlight", help="comma list of part ids to emphasize; the rest are ghosted")
    p.add_argument("--hide", help="comma list of part ids to leave out")
    p.add_argument("--upto", help="show only the first N parts, or parts through id X")
    p.add_argument("--no-grid", action="store_true", help="omit the 1-stud ground grid")
    p.add_argument("--no-edges", action="store_true", help="omit part edge lines")
    p.add_argument("--html", help="also write a standalone interactive viewer page")
    p.set_defaults(func=cmd_view)

    args = ap.parse_args()
    if getattr(args, "extra", None) and args.extra[0] == "--":
        args.extra = args.extra[1:]
    args.func(args)


if __name__ == "__main__":
    main()
