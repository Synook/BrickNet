# BrickNet fork: hand-authored LEGO models

We build LEGO models by writing BrickNet path text directly. The fine-tuned model is not used:
`scripts/generate.py`, `scripts/train.py`, and `eval/` are upstream code we leave alone.

**Before building a model, read `docs/authoring.md`**: workflow, path-text cheat sheet, geometry
conventions (LDU, −y up, yaw is relative to the parent), recipes, and pitfalls.

## Environment

`.claude/hooks/session-start.sh` builds `.venv` (on `PATH`) and downloads the colliders and part meshes
(needs network access to `keeper.mpdl.mpg.de`). If `scripts/author.py` fails for missing modules,
colliders, or meshes, run the hook by hand:
`CLAUDE_CODE_REMOTE=true CLAUDE_PROJECT_DIR=$PWD .claude/hooks/session-start.sh`.

## Tools: `python scripts/author.py <command>`

- `find`, `gallery`, `part [--png]`: find parts and read their connector letters
- `check [--conns ids] [--last N]`: parse errors, placements, links, collisions (exit 1 on any)
- `try FILE NODE EDGE`: compare `{a,b}` variants of the next part before committing to one
- `view FILE OUT.png [--labels] [--conns ids] [--views ...]`: WebGL preview, seconds; read the PNG
- `render FILE OUTDIR`: Blender beauty shot, minutes; for final images only
- `publish FILE GALLERY [--image IMG]`: add a finished model to the gallery with a preview and index

Large, regular models (architecture): write a layout script and convert it with
`python scripts/layout.py ../bricknet-models/<name>/layout.py` (see "Layout scripts" in the guide).

## Conventions

- Models live in the private gallery repo `Synook/bricknet-models`, not here. Attach it (`add_repo`,
  push access), clone it to `../bricknet-models`, draft in `<name>/model.txt` there (title and
  `# prompt:` comment lines first), and `publish` when done. Commit and push the gallery straight to
  `main`; the owner approved direct pushes there. A model is done when `check` reports 0 colliding pairs
  and `view` looks right.
- Python: `black --line-length=120` and `flake8 --max-line-length=120` (versions pinned in `.venv`).
  JavaScript in `scripts/viewer/`: `node --check scripts/viewer/viewer.js`.
- Scratch images and test drafts go in the session scratchpad, not the repo.
