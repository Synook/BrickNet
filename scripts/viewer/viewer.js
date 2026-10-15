// WebGL preview renderer for BrickNet models: flat-shaded parts with edge lines, a stud grid, and
// labels, rendered as a grid of camera views. Driven headlessly by driver.py (scripts/author.py view)
// or interactively from index.html.
//
// Coordinates follow LDraw: +x right, -y up, and the front of a model faces -z. Views are named or
// given as azimuth/elevation in degrees: azimuth 0 looks at the front, 90 at the right side;
// elevation 90 looks straight down.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const PRESETS = {
  front: [0, 0], back: [180, 0], right: [90, 0], left: [270, 0],
  top: [0, 90], bottom: [0, -90], iso: [35, 30], 'iso-back': [215, 30],
};

const BACKGROUND = '#f2f2ef';
// bricknet-render's GLBs store part geometry with z negated relative to LDraw part coordinates
const GLB_TO_LDRAW = new THREE.Matrix4().makeScale(1, 1, -1);
const GHOST = '#c8c8c8';
const MARGIN = 1.12;
const FOV = 30;

export function parseView(spec) {
  if (spec in PRESETS) return { name: spec, az: PRESETS[spec][0], el: PRESETS[spec][1] };
  const m = /^(-?[\d.]+)\/(-?[\d.]+)$/.exec(spec);
  if (!m) throw new Error(`bad view ${spec}: use ${Object.keys(PRESETS).join(', ')} or AZ/EL`);
  return { name: `az ${m[1]} el ${m[2]}`, az: +m[1], el: +m[2] };
}

// A camera basis for azimuth/elevation in three.js space (LDraw with y and z negated).
function basis(az, el) {
  const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el);
  const h = new THREE.Vector3(Math.sin(a), 0, Math.cos(a));
  const back = h.clone().multiplyScalar(Math.cos(e)).add(new THREE.Vector3(0, Math.sin(e), 0)).normalize();
  // straight up or down, screen-up continues the y-up convention toward the far (or near) side
  const upHint = Math.abs(el) > 89.9 ? h.clone().multiplyScalar(-Math.sign(el)) : new THREE.Vector3(0, 1, 0);
  const right = new THREE.Vector3().crossVectors(upHint, back).normalize();
  const up = new THREE.Vector3().crossVectors(back, right).normalize();
  return { right, up, back };
}

// Label positions [id, x, y, width]: smallest parts first, each at its candidate point nearest the
// centroid that keeps clear of labels already placed; a part with no clear point goes unlabeled.
export function placeLabels(picks, width, height) {
  const placed = [], out = [];
  const entries = Object.entries(picks).filter(([, [n]]) => n >= 12).sort((a, b) => a[1][0] - b[1][0]);
  for (const [id, [, cand]] of entries) {
    const w = width(id);
    const spot = cand.find(([x, y]) => placed.every((r) => Math.abs(r[0] - x) * 2 >= r[2] + w + 2 || Math.abs(r[1] - y) >= height + 1));
    if (!spot) continue;
    placed.push([spot[0], spot[1], w]);
    out.push([id, spot[0], spot[1], w]);
  }
  return out;
}

// Connector markers: a dot per marker (hollow when occluded, grey when dim) and a label placed beside it,
// clear of other labels; visible markers claim space first.
function drawMarkers(ctx, marks, x0, y0) {
  ctx.font = '600 11px ui-monospace, monospace';
  const placed = [];
  const order = [...marks].sort((a, b) => (a.hidden - b.hidden) || (a.dim - b.dim));
  for (const m of order) {
    const color = m.dim ? '#8c8c8c' : m.color;
    ctx.beginPath();
    ctx.arc(x0 + m.x, y0 + m.y, m.hidden ? 3 : 4, 0, 2 * Math.PI);
    ctx.lineWidth = 1.5;
    if (m.hidden) {
      ctx.strokeStyle = color;
      ctx.stroke();
    } else {
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.stroke();
    }
  }
  for (const m of order) {
    const w = ctx.measureText(m.label).width + 6, h = 14;
    const spots = [[6, -h - 2], [6, 2], [-w - 6, -h - 2], [-w - 6, 2], [6, -h - 14], [-w - 6, 14], [-w / 2, -h - 10], [-w / 2, 8]];
    const spot = spots.map(([dx, dy]) => [m.x + dx, m.y + dy])
      .find(([x, y]) => placed.every(([px, py, pw]) => x + w <= px || px + pw <= x || y + h <= py || py + h <= y));
    if (!spot) continue;
    placed.push([spot[0], spot[1], w]);
    ctx.globalAlpha = m.hidden ? 0.55 : 1;
    ctx.fillStyle = 'rgba(255,255,255,0.88)';
    ctx.fillRect(x0 + spot[0], y0 + spot[1], w, h);
    ctx.fillStyle = m.dim ? '#666' : m.color;
    ctx.fillText(m.label, x0 + spot[0] + 3, y0 + spot[1] + 11);
    ctx.globalAlpha = 1;
  }
}

export class Viewer {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(1);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND);
    this.root = new THREE.Group();
    this.root.scale.set(1, -1, -1); // LDraw -> three: y up, front toward +z
    this.scene.add(this.root);
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#8a8a80', 1.6));
    this.light = new THREE.DirectionalLight('#ffffff', 1.9);
    this.scene.add(this.light, this.light.target);
    this.persp = new THREE.PerspectiveCamera(FOV, 1, 1, 1e6);
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e6, 1e6);
    this.templates = new Map();
    this.parts = [];
  }

  // model: {parts: [{id, stem, color, matrix: 12 LDR numbers (x y z a..i)}], colors: {code: [hex, alpha, edge]}}
  // getMesh(stem) -> Promise<ArrayBuffer | null>
  async load(model, getMesh) {
    this.colors = model.colors;
    const loader = new GLTFLoader();
    const missing = [];
    for (const stem of new Set(model.parts.map((p) => p.stem))) {
      const buf = await getMesh(stem);
      if (!buf) { missing.push(stem); continue; }
      const gltf = await loader.parseAsync(buf, '');
      const prims = [];
      gltf.scene.traverse((o) => {
        if (!o.isMesh) return;
        const code = +(/color_(\d+)/.exec(o.material.name)?.[1] ?? 16);
        prims.push({ geometry: o.geometry, edges: new THREE.EdgesGeometry(o.geometry, 30), code });
      });
      this.templates.set(stem, prims);
    }
    for (const p of model.parts) {
      const prims = this.templates.get(p.stem);
      if (!prims) continue;
      const [x, y, z, a, b, c, d, e, f, g, h, i] = p.matrix;
      const obj = new THREE.Group();
      obj.matrixAutoUpdate = false;
      obj.matrix.set(a, b, c, x, d, e, f, y, g, h, i, z, 0, 0, 0, 1).multiply(GLB_TO_LDRAW);
      const meshes = [], lines = [];
      for (const prim of prims) {
        const code = prim.code === 16 ? p.color : prim.code === 24 ? null : prim.code;
        const [hex, alpha, edge] = this.colors[code] ?? this.colors[p.color] ?? ['#888888', 1, '#333333'];
        const mesh = new THREE.Mesh(prim.geometry, this.material(hex, alpha));
        const line = new THREE.LineSegments(prim.edges, new THREE.LineBasicMaterial({ color: edge }));
        mesh.userData = { hex, alpha };
        line.userData = { edge };
        obj.add(mesh, line);
        meshes.push(mesh);
        lines.push(line);
      }
      this.root.add(obj);
      this.parts.push({ id: p.id, index: this.parts.length, obj, meshes, lines });
    }
    this.root.updateMatrixWorld(true);
    for (const part of this.parts) part.box = new THREE.Box3().setFromObject(part.obj);
    return { missing };
  }

  material(hex, alpha) {
    return new THREE.MeshLambertMaterial({
      color: hex, flatShading: true, transparent: alpha < 1, opacity: alpha, depthWrite: alpha >= 1,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
  }

  // Visibility and emphasis: show (only these ids), hide (ids), upto (count), highlight (ids, others
  // ghosted), edges (bool).
  configure({ show = null, hide = [], upto = null, highlight = [], edges = true } = {}) {
    const shown = show && new Set(show), hidden = new Set(hide), lit = new Set(highlight);
    for (const part of this.parts) {
      part.visible = (!shown || shown.has(part.id)) && !hidden.has(part.id) && (upto === null || part.index < upto);
      part.ghost = lit.size > 0 && !lit.has(part.id);
      part.obj.visible = part.visible;
      for (const m of part.meshes) {
        const { hex, alpha } = m.userData;
        m.material.color.set(part.ghost ? GHOST : hex);
        m.material.opacity = part.ghost ? 0.18 : alpha;
        m.material.transparent = part.ghost || alpha < 1;
        m.material.depthWrite = !m.material.transparent;
      }
      for (const l of part.lines) {
        l.visible = edges;
        l.material.color.set(part.ghost ? '#999999' : l.userData.edge);
        l.material.transparent = part.ghost;
        l.material.opacity = part.ghost ? 0.12 : 1;
      }
    }
  }

  visibleBox() {
    const box = new THREE.Box3();
    for (const part of this.parts) if (part.visible) box.union(part.box);
    return box;
  }

  partBox(id) {
    const part = this.parts.find((p) => p.id === id);
    if (!part) throw new Error(`no part ${id}`);
    return part.box;
  }

  // Place the camera for one view. target: three-space point or null (model center).
  frame(view, { zoom = 1, pan = [0, 0], target = null, ortho = false, aspect = 1 } = {}) {
    const box = this.visibleBox();
    const center = target ? target.clone() : box.getCenter(new THREE.Vector3());
    const { right, up, back } = basis(view.az, view.el);
    // fit the box as seen along this view
    let hw = 1, hh = 1, depth = 0;
    for (let k = 0; k < 8; k++) {
      const c = new THREE.Vector3(k & 1 ? box.max.x : box.min.x, k & 2 ? box.max.y : box.min.y, k & 4 ? box.max.z : box.min.z);
      c.sub(center);
      hw = Math.max(hw, Math.abs(c.dot(right)));
      hh = Math.max(hh, Math.abs(c.dot(up)));
      depth = Math.max(depth, c.dot(back));
    }
    const half = (Math.max(hw / aspect, hh) * MARGIN) / zoom;
    center.addScaledVector(right, pan[0] * 2 * half * aspect).addScaledVector(up, pan[1] * 2 * half);
    let cam;
    if (ortho) {
      cam = this.ortho;
      Object.assign(cam, { left: -half * aspect, right: half * aspect, top: half, bottom: -half });
      cam.position.copy(center).addScaledVector(back, depth + 1000);
    } else {
      cam = this.persp;
      cam.aspect = aspect;
      const dist = half / Math.tan(THREE.MathUtils.degToRad(FOV / 2)) + depth / zoom;
      cam.position.copy(center).addScaledVector(back, dist);
      cam.near = Math.max(1, dist / 100);
      cam.far = dist + 20 * (depth + half) + 1000;
    }
    cam.up.copy(up);
    cam.lookAt(center);
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
    // key light above-left of the camera, so faces stay distinguishable in every view
    this.light.position.copy(cam.position).addScaledVector(up, half * 2).addScaledVector(right, -half);
    this.light.target.position.copy(center);
    this.light.target.updateMatrixWorld();
    return cam;
  }

  // A stud grid (20 LDU) under the visible parts, aligned to multiples of 20 LDU.
  setGrid(on) {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
    }
    this.grid = null;
    if (!on) return;
    const box = this.visibleBox();
    if (box.isEmpty()) return;
    const span = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) + 120;
    const size = Math.ceil(span / 40) * 40;
    this.grid = new THREE.GridHelper(size, size / 20, '#a9a9a2', '#d3d3cc');
    this.grid.position.set(Math.round(((box.min.x + box.max.x) / 2) / 20) * 20, box.min.y - 0.5, Math.round(((box.min.z + box.max.z) / 2) / 20) * 20);
    this.scene.add(this.grid);
  }

  // Which visible parts own which pixels of one view: {id: [pixel count, label x, label y]}.
  pick(cam, w, h = w) {
    const rt = new THREE.WebGLRenderTarget(w, h);
    const swaps = [];
    for (const part of this.parts) {
      for (const m of part.meshes) {
        swaps.push([m, m.material]);
        const id = part.index + 1;
        m.material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
        m.material.color.setRGB((id >> 16 & 255) / 255, (id >> 8 & 255) / 255, (id & 255) / 255, THREE.LinearSRGBColorSpace);
        m.visible = part.visible && !part.ghost;
      }
      for (const l of part.lines) { swaps.push([l, l.visible]); l.visible = false; }
    }
    const grid = this.grid?.visible, bg = this.scene.background;
    if (this.grid) this.grid.visible = false;
    this.scene.background = new THREE.Color(0, 0, 0);
    this.renderer.setRenderTarget(rt);
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, w, h);
    this.renderer.render(this.scene, cam);
    const px = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, w, h, px);
    this.renderer.setRenderTarget(null);
    rt.dispose();
    for (const [o, v] of swaps) {
      if (o.isMesh) { o.material.dispose(); o.material = v; o.visible = true; } else o.visible = v;
    }
    if (this.grid) this.grid.visible = grid;
    this.scene.background = bg;

    const acc = new Map();
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const k = (y * w + x) * 4;
        const id = (px[k] << 16) | (px[k + 1] << 8) | px[k + 2];
        if (!id) continue;
        const s = acc.get(id) ?? [0, 0, 0, []];
        s[0]++; s[1] += x; s[2] += h - 1 - y;
        if ((x + y) % 3 === 0) s[3].push(x, h - 1 - y);
        acc.set(id, s);
      }
    }
    const out = {};
    for (const [id, [n, sx, sy, pts]] of acc) {
      // label candidates: visible pixels of the part, nearest its centroid first
      const cx = sx / n, cy = sy / n, cand = [];
      for (let k = 0; k < pts.length; k += 2) cand.push([(pts[k] - cx) ** 2 + (pts[k + 1] - cy) ** 2, pts[k], pts[k + 1]]);
      cand.sort((a, b) => a[0] - b[0]);
      out[this.parts[id - 1].id] = [n, cand.slice(0, 600).map((c) => [c[1], c[2]])];
    }
    return out;
  }

  // Render a cols-wide grid of size x size tiles; returns {png: data URL, views: [{name, visible}]}.
  // Tiles default to one per view; explicit tiles are [{view, show: [ids], caption}], e.g. one part each.
  // markers: [{pos: LDraw [x, y, z], label, color, dim}] are drawn on every tile, hollow where occluded.
  // format: 'png' or 'jpeg' (the data URL is still returned as `png`).
  shoot({ views = ['iso'], tiles = null, cols = null, size = 512, zoom = 1, pan = [0, 0], target = null,
    ortho = false, labels = false, grid = true, markers = [], format = 'png', ...vis } = {}) {
    tiles = (tiles ?? views.map((view) => ({ view }))).map((t) => ({ ...t, view: parseView(t.view ?? 'iso') }));
    cols = cols ?? Math.ceil(Math.sqrt(tiles.length));
    const rows = Math.ceil(tiles.length / cols);
    const W = cols * size, H = rows * size;
    this.renderer.setSize(W, H, false);
    let tgt = null;
    if (typeof target === 'string') tgt = this.partBox(target).getCenter(new THREE.Vector3());
    else if (Array.isArray(target)) tgt = new THREE.Vector3(target[0], -target[1], -target[2]);

    const out = document.createElement('canvas');
    out.width = W; out.height = H;
    const ctx = out.getContext('2d');
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, W, H);
    const results = [];
    tiles.forEach((tile, k) => {
      const { view } = tile;
      const col = k % cols, row = Math.floor(k / cols);
      this.configure({ ...vis, show: tile.show ?? null });
      this.setGrid(grid);
      const cam = this.frame(view, { zoom, pan, target: tgt, ortho });
      if (this.grid) this.grid.visible = view.el >= 0;
      const picks = this.pick(cam, size);
      this.renderer.setScissorTest(true);
      this.renderer.setViewport(col * size, (rows - 1 - row) * size, size, size);
      this.renderer.setScissor(col * size, (rows - 1 - row) * size, size, size);
      this.renderer.render(this.scene, cam);
      const marks = markers.length ? this.projectMarkers(cam, markers, size) : [];
      results.push({ tile, cam: cam.clone(), picks, marks, x0: col * size, y0: row * size });
    });
    ctx.drawImage(this.renderer.domElement, 0, 0);
    for (const { tile, cam, picks, marks, x0, y0 } of results) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, y0, size, size);
      ctx.clip();
      if (labels) this.drawLabels(ctx, picks, x0, y0, size);
      if (marks.length) drawMarkers(ctx, marks, x0, y0);
      this.drawAxes(ctx, cam, x0 + 34, y0 + size - 34);
      ctx.font = '600 14px sans-serif';
      ctx.fillStyle = '#333';
      const name = `${tile.view.name}${ortho ? ' ortho' : ''}${zoom !== 1 ? ` x${zoom}` : ''}`;
      ctx.fillText(tile.caption ?? name, x0 + 8, y0 + 18, size - 16);
      ctx.restore();
      ctx.strokeStyle = '#999';
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, size - 1, size - 1);
    }
    return {
      png: format === 'jpeg' ? out.toDataURL('image/jpeg', 0.88) : out.toDataURL('image/png'),
      views: results.map(({ tile, picks }) => ({ name: tile.caption ?? tile.view.name, visible: Object.keys(picks) })),
    };
  }

  // Tile pixel positions of markers, flagging those hidden behind visible geometry.
  projectMarkers(cam, markers, size) {
    const meshes = this.parts.filter((p) => p.visible && !p.ghost).flatMap((p) => p.meshes);
    const sides = meshes.map((m) => m.material.side);
    for (const m of meshes) m.material.side = THREE.DoubleSide; // mirrored placements flip the winding
    const ray = new THREE.Raycaster();
    const out = [];
    for (const mk of markers) {
      const p = new THREE.Vector3(mk.pos[0], -mk.pos[1], -mk.pos[2]);
      const ndc = p.clone().project(cam);
      if (Math.abs(ndc.x) > 1 || Math.abs(ndc.y) > 1) continue;
      const dir = cam.isOrthographicCamera
        ? new THREE.Vector3(0, 0, -1).transformDirection(cam.matrixWorld)
        : p.clone().sub(cam.position).normalize();
      const origin = cam.isOrthographicCamera ? p.clone().addScaledVector(dir, -1e5) : cam.position.clone();
      ray.set(origin, dir);
      const hit = ray.intersectObjects(meshes, false)[0];
      const hidden = !!hit && hit.distance < origin.distanceTo(p) - 6; // a stud occludes its own base by 4
      out.push({ ...mk, x: ((ndc.x + 1) / 2) * size, y: ((1 - ndc.y) / 2) * size, hidden });
    }
    meshes.forEach((m, i) => { m.material.side = sides[i]; });
    return out;
  }

  drawLabels(ctx, picks, x0, y0, size) {
    const font = Math.max(12, Math.round(size / 40));
    ctx.font = `700 ${font}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const [id, px, py, w] of placeLabels(picks, (id) => ctx.measureText(id).width + 6, font + 4)) {
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillRect(x0 + px - w / 2, y0 + py - font / 2 - 2, w, font + 4);
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + px - w / 2, y0 + py - font / 2 - 2, w, font + 4);
      ctx.fillStyle = '#111';
      ctx.fillText(id, x0 + px, y0 + py + 1);
    }
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }

  // LDraw axes as seen from this camera: +x, up (-y), +z (toward the back).
  drawAxes(ctx, cam, cx, cy) {
    const axes = [['+x', [1, 0, 0], '#c0392b'], ['up', [0, 1, 0], '#27ae60'], ['+z', [0, 0, -1], '#2e6fd8']];
    const inv = cam.matrixWorldInverse;
    ctx.lineWidth = 2;
    ctx.font = '600 11px sans-serif';
    for (const [name, v, color] of axes) {
      const d = new THREE.Vector3(...v).transformDirection(inv);
      if (Math.hypot(d.x, d.y) < 0.15) {
        ctx.fillStyle = color;
        ctx.fillText(d.z > 0 ? `${name}⊙` : `${name}⊗`, cx - 8, cy + 4);
        continue;
      }
      const ex = cx + d.x * 24, ey = cy - d.y * 24;
      ctx.strokeStyle = color;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(name, ex + d.x * 8 - 6, ey - d.y * 8 + 4);
    }
  }
}
