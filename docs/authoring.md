# Hand-authoring BrickNet models

How to build a LEGO model by writing BrickNet path text directly, with `scripts/author.py` for lookup,
checking, and pictures. Everything here was learned building the fantasy horse-drawn carriage (76 parts,
`carriage/model.txt` in the gallery repo); read it alongside this guide for a worked example.

## Setup

In Claude Code cloud sessions `.claude/hooks/session-start.sh` prepares everything: a `.venv` (on
`PATH`, so `python` is the venv's) with this package, pybullet, Playwright, bricknet-render + bpy, and the
pinned linters; three.js under `scripts/viewer/node_modules`; and the collision colliders and part meshes
in the BrickNet data dir (`~/.local/share/bricknet/{cvx,glb}`). The mesh downloads need network access to
`keeper.mpdl.mpg.de`. Run the hook by hand (`CLAUDE_CODE_REMOTE=true CLAUDE_PROJECT_DIR=$PWD
.claude/hooks/session-start.sh`) if something is missing; it skips finished steps.

Importing `bricknet` prints a `pybullet build time: ...` banner on stdout; filter it when piping.

## Where models live

Models go in the private gallery repo [Synook/bricknet-models](https://github.com/Synook/bricknet-models),
not here; this repo holds the tools. Attach it to the session (`add_repo`, push access) and clone it
next to this one, at `../bricknet-models`. Each model is a folder:

- `<name>/model.txt`: the commented path text, the source of truth. Its first comment line is the title,
  and a `# prompt: ...` line records what was asked for.
- `<name>/preview.jpg`: a small preview (about 30 KB), written by `publish`.

`.ldr` files and renders are derived; rebuild them when needed instead of committing them. The owner
has approved committing and pushing straight to the gallery's `main`, with no PRs.

## The loop

1. **Plan on paper.** Pick the root part (it sits at the origin with identity rotation, so its axes
   are the model's), the footprint in studs, the palette, and a height budget in LDU. Work out the
   critical heights first: wheel radius vs. axle height, where a horse's hooves land, what must clear
   what.
2. **Find parts.** `find` searches names (every word must match); `gallery out.png --find "..."` shows the
   hits as pictures; `part NAME` lists connector letters with part-local positions, and
   `part NAME --png out.png` draws them on the part. Check a part's connectors before designing around
   it: decorative parts often lack the connector you would expect.
3. **Write the draft in stages** of 10–20 parts in `M=../bricknet-models/<name>/model.txt`, starting
   with the title and `# prompt:` lines, with `#` comments per section.
4. **After each stage:** `check` (parse errors with line numbers, placement table, overlaps), then
   `view` for a picture. To attach the next part, read where the parent's connectors are with
   `check --conns X` (world positions, `*` marks mated ones) or `view --conns X` (drawn on the model).
5. **Unsure of a letter, yaw, or flip?** Don't edit-and-rerun; `try` places every variant at once:

   ```bash
   python scripts/author.py try $M "animal horse complete | white" "bj stud hole {a,b} open a {90,270}"
   ```

   Each variant reports origin, orientation, the parts it links to, and collisions. `--append K` adds
   variant K to the draft.
6. **Finish:** a labelled `view` grid to confirm, then a beauty shot into the scratchpad, e.g.
   `render $M out/ --views 2 --samples 96 --resolution 1024x768 --yaw0 35 --pitch -20` (about 3 minutes
   per view on 4 CPU cores; extra flags pass through to bricknet-render).
7. **Publish:** `publish $M ../bricknet-models --image out/model_0000.png` refuses a model with
   collisions, writes `preview.jpg` (downscaled from the render; without `--image`, a WebGL view), and
   regenerates the gallery's `README.md` index. Commit and push the gallery to `main`.

Commands at a glance:

| command | use |
| --- | --- |
| `find WORDS` | search part names; shows connector counts |
| `gallery OUT [PARTS] [--find WORDS]` | contact sheet, one tile per part |
| `part NAME [--png OUT] [--sub stud,hole]` | connector letters and local positions (and a labelled render) |
| `check FILE [--conns ids] [--last N] [-o LDR] [--path TXT]` | parse, place, collide; exits 1 on errors or overlaps |
| `try FILE NODE EDGE [--append K]` | compare `{a,b}` variants of the next part |
| `view FILE OUT [--views ...] [--labels] [--conns ids] ...` | WebGL preview grid, ~5 s |
| `view FILE --html OUT` | standalone interactive viewer page |
| `render FILE OUTDIR [...]` | Blender render via bricknet-render, slow |
| `publish FILE GALLERY [--image IMG]` | add a model to the gallery checkout with a preview; update its index |
| `index GALLERY` | regenerate the gallery's `README.md` |

## Path text essentials

```
a plate 4x10 | pearl gold                 node: ID NAME | COLOR
b technic brick 1x2 with hole | pearl gold
a stud hole af open a 0                   edge for b: parent a's hole af takes b's open stud a, yaw 0
```

- Every node after the first is followed by its edge line. IDs are positional (`a`…`z`, `aa`, …): the
  parser ignores the label and edges resolve by position, so `check` rejects a mislabeled node.
- Names and colors must match `part_names.json` / `color_names.json` exactly; `find` to confirm.
- In an edge the parent's connector comes first, then the child's:

  | family | form | mating pairs |
  | --- | --- | --- |
  | stud | `P stud PSUB PCONN CSUB CCONN YAW` | stud–hole, stud–tube, open–hole, open–post, open–tube |
  | axle | `P axle PSUB PCONN CSUB CCONN FLIP YAW SLIDE` | axle–cross, axle–socket, bar–clip, bar–cross, pin–socket |
  | hinge | `P hinge SUB POL PCONN CCONN FLIP YAW` | same subtype letter, opposite polarity |
  | ball | `P ball SUB POL PCONN CCONN RX RY RZ` | same subtype letter, opposite polarity |
  | fixed | `P fixed SUB POL PCONN CCONN` | same subtype letter, opposite polarity |

  For hinge/ball/fixed, `POL` is the parent's polarity; `part` prints e.g. `hinge h on` and names the
  other side (`h in`).
- Connector letters count within one (subtype, polarity) group, ordered by position (x, then z, ...).
  Never guess them; read `part`, `check --conns`, or `view --conns`.
- `stud` = solid stud, `open` = hollow stud (usually with a `clip` connector at the same spot, which grips
  bars), `hole` = anti-stud under a stud position, `tube` = anti-stud between studs (offset placement),
  `post` = the bar under 1-wide bricks.

## Geometry conventions

- LDraw units (LDU): 20 per stud pitch, 8 per plate, 24 per brick, 4 per stud height. **−y is up.**
  LDraw's front faces −z; the viewer's `front` view looks at the −z face, `right` at the +x face.
- Most bricks and plates have their origin at the top-surface center: studs at `y=0`, holes at
  `y=height`. Windows and some special parts differ (`part` shows bounds and positions).
- Grid alignment: with an even-sized root at the origin, every stud sits at odd multiples of 10. A part
  only seats if all its studs land on that grid; `check`'s **links** column counts neighbors actually
  mated (tree edge or not), so a plate showing fewer links than expected is misaligned or floating.
- **Stud-joint yaw is relative to the parent**, not the world: child yaw = parent yaw + edge yaw for
  upright parts. Blocks hung under a spacer plate that itself has yaw 90 needed edge yaw 270 to come
  out at world yaw 0. `check`'s orientation column shows the result.
- Yaw direction: a child-local offset `(x, z)` lands at `(x cos θ + z sin θ, −x sin θ + z cos θ)` from
  the joint, so yaw 90 sends the child's +x to −z and +z to +x.
- Hinge and axle `FLIP`/`YAW` combinations are easiest to settle with `try` (four variants: regular/flip
  × the two yaws that keep the part level).
- `check` allows 1 LDU of overlap per pair (the collider tolerance); fixed-edge children may overlap
  their parent.

## Recipes

**Spoked wagon wheels** (`wheel 3.2x56 with 10 spokes wooden`, radius 70; `wheel 3.2x43 ...`,
radius 54): a `technic brick 1x2 with hole` under the chassis (socket along its local z), a
`technic pin` in its socket (`axle socket a pin a regular 0 0` on one side, `flip` on the other so it
points outward), the wheel on the pin with `axle pin b socket a flip 0 0` (hub against the brick, rim
dished outward). Axle height = wheel radius, so put the small wheels' blocks on extra plates (two
plates, 16 LDU, for 70 vs 54) to keep the chassis level.

**Car wheels:** `plate, modified 2x2 with wheels holder` bars + a wheel rim's clip
(`axle bar a clip a regular 90 0`); see the car example (`python -m bricknet sample
src/bricknet/_data/car.ldr`).

**Horse and hitch:** `hinge plate 1x2 with 3 fingers on side` (fingers on its local −z side, hinge 4 below
its top) → `animal horse hitching with hinge` with `hinge h on a a flip 270` (level, pointing −x when the
hinge plate has yaw 90) → `animal horse complete` with `stud hole b open a 90` (head toward −x) →
`animal horse saddle with two clips` on the hitching (`stud stud b hole a 90`) → `minifig plume small`
in the saddle clips (`axle clip a bar a regular 0 0`). Hooves sit 57 below the horse's back studs and
the hitching's holes 13 above its hinge, so put the hinge 44 above the ground. `animal horse barding`
collides with the hitching's side rails; the winged head armour has no connector the horse can take.

**Dome roof:** `dish 4x4 inverted` mates only through its center tube, so it needs a stud exactly at
its center, at the surface its rim rests on. Use a `plate 2x2 round with 1 centre stud` jumper and fill
the rest of that layer to the same height. Anything under the dish must be a tile: studs there
collide (up to 3.6 LDU). Crown it with `minifig crown with 5 points` on the dish's top stud.

**Round pillars:** stack `brick 1x1 round with hollow stud` (`stud stud X tube a`, then
`stud open a tube a`).

**Windows:** `window 1x4x3 curved top with 4 panes and sill` has its origin at the bottom and no top
studs, so carry a roof on pillars beside it; `window 1x2x3 without sill w trans-clear glass complete` has
its origin at the top (open studs at y=0, holes at y=72).

**Finishing:** cover exposed studs with tiles (`tile 1x4 with groove`, `tile 2x2 with groove`,
`tile 2x4`); keep plates where something still has to attach (finials, lanterns).

## Pitfalls

- Perspective views distort alignment; judge it with `--ortho` front/top/right, zoomed on a part
  (`--target X --zoom 3`), with `--highlight` to ghost the rest.
- `view --labels` lists parts not visible from any requested view: buried parts are fine, but a part you
  expected to see there is usually misplaced.
- bricknet-render's GLB meshes are stored z-mirrored; the viewer undoes it per part. If a part ever
  looks inside out in `view`, compare with `render` before trusting either.
- `find` matches every word as a substring and ranks exact, then prefix, then shorter names first, so
  the part you want can sit below the shown hits. Add a distinguishing word (names mostly use
  British spellings: `centre`, `grey`, `colour`) or raise `--limit`.
- Translucent colors (`trans light blue`, ...) render see-through in both renderers; `pearl gold`,
  `medium lavender`, `white`, and `bright pink` made a good fantasy palette.
