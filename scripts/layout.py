#!/usr/bin/env python3
"""Lay out a model on the stud grid, then convert it to path text with bricknet's own connector finder.

A layout script is plain Python run by this file, with a `Layout` named `L` in scope:

    python scripts/layout.py ../bricknet-models/<name>/layout.py   # writes model.txt beside it

    TITLE = "Office block"
    PROMPT = "a six-storey office building"
    L.section("base")
    L.add("plate 6x24", "dark bluish grey", i=-2, j=-3, level=-1)
    L.ring(0, 0, 8, 9, level=0, kind="brick", color="white")
    L.fill(0, 0, 8, 9, level=13, kind="plate", color="medium lavender")

Grid: stud (i, j) covers LDU x in [-20i - 20, -20i] and z in [20j, 20j + 20]: i grows east, toward -x, and
j grows south, toward +z, so the model's front (LDraw -z) faces north and the layout is not mirrored (with
-y up, east must be -x when north is -z; looking south from the front, east is on the left). The viewer's
"right" view therefore shows the west face. `level` counts plates above y = 0 (-y is up); a
part sits on its level and rises by its own height (plate 1, brick 3). Footprints come from the
colliders, so parts are placed by their bounding rectangle, rotated by `rot` (0 or 90).

Conversion: parse_ldr finds every mated connector pair; the tree then attaches parts in authoring order,
each to its earliest-placed neighbor, so `section` labels become comments in the path text. Parts that
touch nothing are reported, and the result should still go through `author.py check` for collisions.
"""

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from bricknet import collision
from bricknet.data import load_catalog
from bricknet.graph import _adjacency, _build_tree, parse_ldr
from bricknet.tree import serialize_tree

_LENGTHS = {
    "plate": (12, 10, 8, 6, 4, 3, 2, 1),
    "brick": (16, 12, 10, 8, 6, 4, 3, 2, 1),
    "tile": (8, 6, 4, 3, 2, 1),
}
_SIZES = (24, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1)


@lru_cache(maxsize=None)
def dims(name):
    """(studs along local x, studs along local z, height in LDU, local footprint center x, z)."""
    lo, hi = collision._bounds(load_catalog().name_to_id[name])
    return (
        round((hi[0] - lo[0]) / 20),
        round((hi[2] - lo[2]) / 20),
        round(hi[1]),
        (lo[0] + hi[0]) / 2,
        (lo[2] + hi[2]) / 2,
    )


@lru_cache(maxsize=None)
def piece(kind, along_i, along_j):
    """(name, rot) of a `kind` part covering along_i x along_j studs, or None."""
    names = load_catalog().name_to_id
    a, b = sorted((along_i, along_j))
    for name in (f"{kind} {a}x{b}", f"{kind} {a}x{b} with groove"):
        if name in names:
            w, d = dims(name)[:2]
            if (w, d) == (along_i, along_j):
                return name, 0
            if (d, w) == (along_i, along_j):
                return name, 90
    return None


def _ry(deg):
    c, s = round(math.cos(math.radians(deg))), round(math.sin(math.radians(deg)))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def split(length, kind, phase=0):
    """Piece lengths covering `length` studs; a nonzero phase starts with a short piece so stacked
    runs stagger their joints."""
    avail = [n for n in _LENGTHS[kind] if piece(kind, n, 1)]
    out, left = [], length
    if phase and length > phase:
        first = max(n for n in avail if n <= phase)
        out.append(first)
        left -= first
    while left:
        n = max(n for n in avail if n <= left)
        out.append(n)
        left -= n
    return out


def _lines(cells, axis):
    """Maximal straight runs (i, j, length) of a set of studs along axis "i" or "j"."""
    out = []
    for i, j in sorted(cells, key=lambda c: (c[1], c[0]) if axis == "i" else c):
        prev = (i - 1, j) if axis == "i" else (i, j - 1)
        if prev in cells:
            continue
        n = 1
        while ((i + n, j) if axis == "i" else (i, j + n)) in cells:
            n += 1
        out.append((i, j, n))
    return out


class Layout:
    def __init__(self):
        self.parts = []  # (stem, color code, 4x4 matrix, section)
        self.sec = ""
        cat = load_catalog()
        self._cat = cat

    def section(self, label):
        self.sec = label

    def color(self, c):
        return c if isinstance(c, int) else self._cat.color_to_code[c]

    def add(self, name, color, i, j, level, rot=0, center=False):
        """Place `name` with its footprint's min corner at stud (i, j), bottom at `level`. With center=True
        the part's origin goes on that stud's center instead: for irregular parts (leaves, round pieces)
        and pieces centered on one stud (a 2x2 leaf cluster on a 1x1 trunk)."""
        w, d, h, cx, cz = dims(name)
        fw, fd = (w, d) if rot % 180 == 0 else (d, w)
        if center:
            fw = fd = 1
            cx = cz = 0.0
        r = _ry(rot)
        center = np.array([-(20 * i + 10 * fw), 0.0, 20 * j + 10 * fd])
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = center - r @ np.array([cx, 0.0, cz])
        m[1, 3] = -8 * level - h
        self.parts.append((self._cat.id_to_stem[self._cat.name_to_id[name]], self.color(color), m, self.sec))
        return len(self.parts) - 1

    def run(self, i, j, length, axis, level, kind, color, phase=0):
        """A 1-wide line of `kind` pieces from stud (i, j) along axis "i" or "j". `color` may be a function
        of the stud (i, j), level -> color; pieces then split wherever the color changes, and studs it maps
        to None stay empty (to leave room for a part that crosses the line)."""
        cells = [(i + k, j) if axis == "i" else (i, j + k) for k in range(length)]
        colors = [color(ci, cj, level) if callable(color) else color for ci, cj in cells]
        pos = 0
        while pos < length:
            end = pos
            while end < length and colors[end] == colors[pos]:
                end += 1
            if colors[pos] is None:
                pos = end
                continue
            for n in split(end - pos, kind, phase if pos == 0 else 0):
                name, rot = piece(kind, n, 1) if axis == "i" else piece(kind, 1, n)
                ci, cj = cells[pos]
                self.add(name, colors[pos], ci, cj, level, rot)
                pos += n

    def ring(self, i0, j0, i1, j1, level, kind, color, phase=0, sides="nesw"):
        """The 1-wide perimeter of [i0, i1) x [j0, j1). Odd phases swap which sides own the corners and
        shift the joints, so alternate layers interlock. `sides` limits it to some of n/e/s/w."""
        own_ns = phase % 2 == 0
        ii0, ii1 = (i0, i1) if own_ns else (i0 + 1, i1 - 1)
        jj0, jj1 = (j0 + 1, j1 - 1) if own_ns else (j0, j1)
        p = 2 if phase % 2 else 0
        if "n" in sides and ii1 > ii0:
            self.run(ii0, j0, ii1 - ii0, "i", level, kind, color, p)
        if "s" in sides and ii1 > ii0:
            self.run(ii0, j1 - 1, ii1 - ii0, "i", level, kind, color, p)
        if "w" in sides and jj1 > jj0:
            self.run(i0, jj0, jj1 - jj0, "j", level, kind, color, p)
        if "e" in sides and jj1 > jj0:
            self.run(i1 - 1, jj0, jj1 - jj0, "j", level, kind, color, p)

    @staticmethod
    def edge(cells):
        """The studs of a footprint (a set of (i, j)) with a side on its boundary."""
        return {(i, j) for i, j in cells if not {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)} <= cells}

    def outline(self, cells, level, kind, color, phase=0):
        """The 1-wide boundary of any footprint, for walls around non-rectangular plans. Even phases lay the
        rows first and odd phases the columns, as `ring` does (a rectangle's outline is its ring), so stacked
        layers interlock."""
        edge = self.edge(set(cells))
        first, second = ("i", "j") if phase % 2 == 0 else ("j", "i")
        singles = set()
        for axis, pool in ((first, edge), (second, singles)):
            for i, j, n in _lines(set(pool), axis):
                if n == 1 and axis == first:
                    singles.add((i, j))
                else:
                    self.run(i, j, n, axis, level, kind, color, 2 if phase % 2 else 0)

    def area(self, cells, level, kind, color):
        """Cover any set of studs with `kind` pieces: each row is cut into spans, and consecutive rows with
        the same span are filled as one rectangle."""
        cells = set(cells)
        rects, open_ = [], {}  # span (i0, i1) -> (first row, last row)
        for j in sorted({j for _, j in cells}) + [None]:
            spans = set() if j is None else {(i, i + n) for i, _, n in _lines({c for c in cells if c[1] == j}, "i")}
            for span, (first, last) in list(open_.items()):
                if span not in spans or last != j - 1:
                    rects.append((span[0], first, span[1], last + 1))
                    del open_[span]
            for span in spans:
                open_[span] = (open_.get(span, (j, j))[0], j)
        for r in rects:
            self.fill(*r, level, kind, color)

    def fill(self, i0, j0, i1, j1, level, kind, color, holes=()):
        """Cover [i0, i1) x [j0, j1), minus `holes` (rectangles (i0, j0, i1, j1)), with `kind` pieces: the
        largest piece that fits the corner, then the strip to its right and the rest below, recursively."""
        if i1 <= i0 or j1 <= j0:
            return
        for k, (hi0, hj0, hi1, hj1) in enumerate(holes):
            if hi0 < i1 and hi1 > i0 and hj0 < j1 and hj1 > j0:  # split around the first overlapping hole
                rest = holes[:k] + holes[k + 1 :]
                for r in (
                    (i0, j0, i1, hj0),
                    (i0, hj1, i1, j1),
                    (i0, max(j0, hj0), hi0, min(j1, hj1)),
                    (hi1, max(j0, hj0), i1, min(j1, hj1)),
                ):
                    self.fill(*r, level, kind, color, rest)
                return
        a, b = max(
            ((a, b) for a in _SIZES if a <= i1 - i0 for b in _SIZES if b <= j1 - j0 and piece(kind, a, b)),
            key=lambda ab: (ab[0] * ab[1], ab[0]),
        )
        name, rot = piece(kind, a, b)
        self.add(name, color, i0, j0, level, rot)
        self.fill(i0 + a, j0, i1, j0 + b, level, kind, color)
        self.fill(i0, j0 + b, i1, j1, level, kind, color)

    def describe(self, indices):
        """'#k name at stud (i, j) level L (section)' for each part index: the stud under its origin."""
        cat, out = self._cat, []
        for k in indices:
            stem, _, m, sec = self.parts[k]
            name = cat.id_to_name[cat.stem_to_id[stem]]
            i, j = math.floor(-m[0, 3] / 20), math.floor(m[2, 3] / 20)
            level = round((-m[1, 3] - dims(name)[2]) / 8)
            out.append(f"#{k} {name} at stud ({i}, {j}) level {level} ({sec})")
        return out

    def ldr(self):
        lines = []
        for stem, color, m, _ in self.parts:
            nums = " ".join(f"{round(v, 6) + 0.0:.12g}" for v in [*m[:3, 3], *m[:3, :3].flat])
            lines.append(f"1 {color} {nums} {stem}.dat")
        return "\n".join(lines) + "\n"

    def to_path(self, root=0):
        """(path text with section comments, indices of parts not connected to the root)."""
        graph = parse_ldr(self.ldr())
        adj = _adjacency(graph.edges)
        placed, order, parent_of = {root}, [root], {}
        pending = sorted(set(range(len(self.parts))) - {root})
        while pending:
            rest = []
            for v in pending:
                link = next(((u, ei, not fwd) for u, ei, fwd in adj.get(v, []) if u in placed), None)
                if link:
                    placed.add(v)
                    order.append(v)
                    parent_of[v] = link
                else:
                    rest.append(v)
            if len(rest) == len(pending):
                break
            pending = rest
        lines = serialize_tree(_build_tree(graph, order, parent_of)).split("\n")
        chunks = [lines[:1]] + [lines[k : k + 2] for k in range(1, len(lines) - 1, 2)]
        out, last = [], None
        for node, chunk in zip(order, chunks):
            sec = self.parts[node][3]
            if sec and sec != last:
                out.append(f"# {sec}")
                last = sec
            out.extend(chunk)
        return "\n".join(out) + "\n", pending


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/layout.py LAYOUT.py  (writes model.txt beside it)")
    script = Path(sys.argv[1])
    env = {"L": Layout(), "TITLE": script.parent.name, "PROMPT": "", "NOTES": []}
    exec(compile(script.read_text(), str(script), "exec"), env)
    text, loose = env["L"].to_path()
    header = [f"# {env['TITLE']}"] + ([f"# prompt: {env['PROMPT']}"] if env["PROMPT"] else [])
    header += [f"# {n}" for n in env["NOTES"]] + [f"# generated by BrickNet scripts/layout.py from {script.name}"]
    out = script.parent / "model.txt"
    out.write_text("\n".join(header) + "\n" + text)
    print(f"{out}: {len(env['L'].parts)} parts, {len(env['L'].parts) - len(loose)} in the tree")
    if loose:
        sys.exit(
            f"{len(loose)} parts touch nothing connected to the root:\n  " + "\n  ".join(env["L"].describe(loose)[:30])
        )


if __name__ == "__main__":
    main()
