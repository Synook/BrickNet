"""Headless driver for the WebGL viewer: LDR text -> PNG of one or more camera views, or a standalone
interactive HTML page.

The page (index.html + viewer.js) is served from this directory, with part meshes from bricknet-render's
GLB library (<BrickNet data dir>/glb), and rendered in headless Chromium through Playwright.
colors.json maps LDraw color codes to [fill, alpha, edge]: LDConfig.ldr, plus the extended BrickNet
codes with values from bricknet-render's material catalog.
"""

import base64
import functools
import http.server
import json
import threading
from pathlib import Path

from bricknet.collision import data_dir
from bricknet.tree import i2id

HERE = Path(__file__).resolve().parent
THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/"


def glb_dir() -> Path:
    return data_dir() / "glb"


def ldr_parts(ldr: str) -> list[dict]:
    """Top-level type-1 lines as viewer parts, labelled a, b, ... in file order."""
    parts = []
    for line in ldr.split("\n"):
        t = line.split()
        if len(t) >= 15 and t[0] == "1":
            stem = " ".join(t[14:]).lower().removesuffix(".dat")
            parts.append(
                {"id": i2id(len(parts)), "stem": stem, "color": int(t[1]), "matrix": list(map(float, t[2:14]))}
            )
    return parts


def _colors() -> dict:
    return json.loads((HERE / "colors.json").read_text())


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves this directory, plus /glb/<stem>.glb from the mesh library."""

    def translate_path(self, path):
        path = path.split("?", 1)[0]
        if path.startswith("/glb/"):
            return str(glb_dir() / Path(path).name)
        return super().translate_path(path)

    def log_message(self, *args):
        pass


class Session:
    """One headless browser with the viewer loaded; reuse it to shoot many images of one model."""

    def __init__(self, parts: list[dict]):
        from playwright.sync_api import sync_playwright

        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(_Handler, directory=str(HERE))
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        self.page = self.browser.new_page()
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        self.page.goto(f"http://127.0.0.1:{self.server.server_address[1]}/index.html")
        self.page.wait_for_function("window.bricknetReady === true")
        self.missing = self._eval(
            """async (model) => {
                window.viewer = new BrickNet.Viewer(document.getElementById('c'));
                const get = async (stem) => {
                    const r = await fetch(`glb/${stem}.glb`);
                    return r.ok ? r.arrayBuffer() : null;
                };
                return (await viewer.load(model, get)).missing;
            }""",
            {"parts": parts, "colors": _colors()},
        )

    def _eval(self, js, arg):
        try:
            return self.page.evaluate(js, arg)
        except Exception as e:
            raise RuntimeError("; ".join(self.errors) or str(e)) from None

    def shoot(self, out: Path, **opts) -> list[dict]:
        """Write a PNG of the requested views; returns [{name, visible: [part ids]}] per view."""
        res = self._eval("(opts) => viewer.shoot(opts)", opts)
        out.write_bytes(base64.b64decode(res["png"].split(",", 1)[1]))
        return res["views"]

    def close(self):
        self.browser.close()
        self.pw.stop()
        self.server.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def export_html(parts: list[dict], out: Path, title: str = "BrickNet Viewer") -> list[str]:
    """Standalone interactive page: meshes embedded, three.js from a CDN. Returns stems without meshes."""
    meshes, missing = {}, []
    for stem in sorted({p["stem"] for p in parts}):
        path = glb_dir() / f"{stem}.glb"
        if path.exists():
            meshes[stem] = base64.b64encode(path.read_bytes()).decode()
        else:
            missing.append(stem)
    model = json.dumps({"parts": parts, "colors": _colors(), "meshes": meshes}, separators=(",", ":"))
    html = (HERE / "index.html").read_text()
    viewer = (
        (HERE / "viewer.js").read_text().replace("export const ", "const ").replace("export function ", "function ")
    )
    viewer = viewer.replace("export class ", "class ")
    html = html.replace("./node_modules/three/build/three.module.js", THREE_CDN + "build/three.module.js")
    html = html.replace("./node_modules/three/examples/jsm/", THREE_CDN + "examples/jsm/")
    html = html.replace("<title>BrickNet Viewer</title>", f"<title>{title}</title>")
    line = next(ln for ln in html.split("\n") if "BRICKNET-INLINE-VIEWER" in ln)
    html = html.replace(line, viewer)
    model = model.replace("</", "<\\/")
    html = html.replace(
        '<script id="model" type="application/json">null</script>',
        f'<script id="model" type="application/json">{model}</script>',
    )
    out.write_text(html)
    return missing
