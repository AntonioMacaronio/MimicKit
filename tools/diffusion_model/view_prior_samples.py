"""Loop generated carry-prior windows next to real held-out windows in Viser.

Reads sample_carry_prior.py output. Each column is one box size in one color (blue,
red, green, yellow, purple, ...): the real window on top in a darker shade, and below
it every window the prior generated for that size, one row per sample, each from its
own noise. Windows are decoded with mimickit_smp.decode and posed with MuJoCo forward
kinematics. A box is drawn in a light tint of its column's color and turns black on
frames where it is lifted (bottom above 10 cm) and neither hand origin is within
8 cm of it (check_carry_samples.py's "hands off box"). Windows have no world pose,
so each is drawn in its own frame (last root XY at the lane origin). Samples of a
skill-conditioned prior are labeled with their skill, and the Skill menu pages
through one skill at a time. Several --samples files (one per prior, e.g. drawn
from the same held-out windows) share one viewer through the Prior menu.

To stay responsive through a share URL, every robot part is one instanced mesh
across all lanes and all boxes are a single instanced unit cube scaled per lane.
A frame is therefore about 80 messages, sent at most --fps times a second and
only while a browser is connected; more per frame stalls the browser behind a
backlog of updates. Robot meshes keep full detail and viser's browser-side LOD
simplifies distant instances; vertex clustering here flipped faces and opened
holes in the G1 meshes.

    source scripts/source_hiphi_retargeting.sh   # in FAR-Holosoma: viser + mujoco
    PYTHONPATH=/path/to/MimicKit:$PYTHONPATH "$FAR_RETARGET_PYTHON" \
        tools/diffusion_model/view_prior_samples.py --samples <run>/eval/samples.npz [<run 2>/...] \
        [--names "prior 1" "prior 2"] --scene <any replay scene.xml with G1 meshes> --share
"""

import argparse
import json
import logging
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser
from scipy.spatial.transform import Rotation

from mimickit_smp.decode import decode_window, matrix_to_wxyz
from mimickit_smp.features import FeatureSchema

COLUMN_SPACING, ROW_SPACING = 1.6, 2.0
# One color per box size (column): generated windows in the color, the real window in a
# darker shade, boxes in a light tint.
COLUMN_COLORS = (
    ("blue", (60, 110, 230)),
    ("red", (215, 55, 55)),
    ("green", (55, 165, 75)),
    ("yellow", (225, 195, 30)),
    ("purple", (145, 75, 200)),
    ("cyan", (40, 180, 190)),
    ("orange", (235, 130, 40)),
    ("pink", (225, 110, 170)),
)
MAX_SAMPLE_ROWS = 8
HANDS_OFF_BOX = (25, 25, 25)
POSE_KEYS = ("root_pos", "root_rot", "joint_pos", "box_pos", "box_rot", "box_extents")  # What posing reads.


def drop_empty_handshakes(record):
    """The share tunnel opens and drops empty connections; websockets logs each with a traceback."""
    error = record.exc_info[1] if record.exc_info else None
    return not (record.getMessage() == "opening handshake failed" and isinstance(getattr(error, "__cause__", None), EOFError))


logging.getLogger("websockets.server").addFilter(drop_empty_handshakes)
CUBE_VERTICES = np.array(
    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], np.float32
)
CUBE_FACES = np.array(  # Counter-clockwise seen from outside.
    [
        [0, 1, 2],
        [1, 3, 2],
        [4, 6, 5],
        [5, 6, 7],
        [0, 4, 1],
        [1, 4, 5],
        [2, 3, 6],
        [3, 7, 6],
        [0, 2, 4],
        [2, 6, 4],
        [1, 5, 3],
        [3, 5, 7],
    ],
    np.uint32,
)


def robot_parts(model):
    """Robot visual meshes (group 1), skipping the scene's own box body."""
    box = model.body("largebox_link").id
    parts = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 1:
            continue
        if model.geom_bodyid[geom] == box:
            continue
        mesh = model.geom_dataid[geom]
        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        f, m = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        parts.append((geom, model.mesh_vert[a : a + n].copy(), model.mesh_face[f : f + m].copy()))
    return parts


def box_distance(points, center, rotation, extents):
    local = (points - center) @ rotation
    q = np.abs(local) - extents / 2
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


def tint(color):
    return tuple(int(0.45 * c + 0.55 * 255) for c in color)


def shade(color):
    return tuple(int(0.55 * c) for c in color)


def lane_colors(rows, cols):
    """Robot and box colors per lane (row-major): one color per column; real rows (sample None) darker."""
    base = [COLUMN_COLORS[c % len(COLUMN_COLORS)][1] for c in range(cols)]
    robot = np.array([shade(base[c]) if sample is None else base[c] for _, sample in rows for c in range(cols)], dtype=np.uint8)
    box = np.array([tint(base[c]) for _ in rows for c in range(cols)], dtype=np.uint8)
    return robot, box


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path, nargs="+", required=True, help="One file per prior")
    parser.add_argument("--names", nargs="*", help="Names in the Prior menu (default: run directory names)")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15.0, help="Maximum scene updates per second")
    parser.add_argument("--port", type=int, default=8692)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    names = args.names or [path.resolve().parent.parent.name for path in args.samples]
    if len(names) != len(args.samples) or len(set(names)) != len(names):
        parser.error("Give one distinct name per samples file")
    files = {name: np.load(path, allow_pickle=False) for name, path in zip(names, args.samples)}
    first = files[names[0]]
    if len({str(f["feature_schema_sha256"]) for f in files.values()}) != 1:
        parser.error("The samples files use different feature schemas")
    index = Path(str(first["index"]))
    schema = FeatureSchema.from_dict(json.loads(index.read_text())["feature_schema"]).current_features()
    generated = {
        n: f["generated_motion"] if f["generated_motion"].ndim == 4 else f["generated_motion"][:, None]  # Older files.
        for n, f in files.items()
    }
    per_size = min(min(g.shape[1] for g in generated.values()), MAX_SAMPLE_ROWS)
    rows = [("real", None)] + [(f"sample {k + 1}", k) for k in range(per_size)]
    def decode(motion, size):
        """Decode in double precision, keep only the pose channels in single (a few priors fit in RAM)."""
        d = decode_window(torch.from_numpy(motion).double(), size, schema)
        return {k: d[k].float() for k in POSE_KEYS}

    priors = {}
    for n, f in files.items():
        size = torch.from_numpy(f["box_size"]).double()
        decoded = {None: decode(f["real_motion"], size)}
        for k in range(per_size):
            decoded[k] = decode(generated[n][:, k], size)
        priors[n] = {
            "size": size,
            "decoded": decoded,
            "sampler": str(f["sampler"]) if "sampler" in f.files else "the prior's sampler",
            # Skill-conditioned samples: each column is labeled with its skill and box size.
            "skills": f["class_names"].tolist() if "class_names" in f.files else None,
            "class_ids": f["class_ids"] if "class_ids" in f.files else None,
        }
    del generated
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    parts = robot_parts(model)
    geom_ids = np.asarray([p[0] for p in parts])
    hands = [model.body(n).id for n in ("left_rubber_hand_link", "right_rubber_hand_link")]
    width = schema.window_frames
    cols = args.columns
    lanes = len(rows) * cols

    def pose(prior, sample, i, t):
        d = prior["decoded"][sample]
        data.qpos[:3] = d["root_pos"][i, t].numpy()
        data.qpos[3:7] = matrix_to_wxyz(d["root_rot"][i, t]).numpy()
        data.qpos[7:36] = d["joint_pos"][i, t].numpy()
        data.qpos[36:39] = [0.0, 0.0, -10.0]  # Hide the scene's own box.
        data.qpos[39:43] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_kinematics(model, data)
        center, rot = d["box_pos"][i, t].numpy(), d["box_rot"][i, t].numpy()
        ext = d["box_extents"][i].numpy()
        bottom = center[2] - 0.5 * np.abs(rot[2]) @ ext
        off = bottom > 0.10 and min(box_distance(data.xpos[h], center, rot, ext) for h in hands) > 0.08
        return data.geom_xpos[geom_ids].copy(), data.geom_xmat[geom_ids].copy(), center, rot, ext, off

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    depth = (len(rows) - 1) * ROW_SPACING
    server.scene.add_grid(
        "/ground",
        width=cols * COLUMN_SPACING + 4,
        height=depth + 4,
        cell_size=0.5,
        position=((cols - 1) * COLUMN_SPACING / 2, -depth / 2, 0.0),
    )
    offsets = np.array(
        [[c * COLUMN_SPACING, -r * ROW_SPACING, 0.0] for r in range(len(rows)) for c in range(cols)]
    )
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (lanes, 1)).astype(np.float32)
    robot_colors, box_colors = lane_colors(rows, cols)
    robot = [
        server.scene.add_batched_meshes_simple(
            f"/robot/{g}",
            vertices=v,
            faces=f,
            batched_wxyzs=identity,
            batched_positions=offsets.astype(np.float32),
            batched_colors=robot_colors,
            side="double",  # A few source triangles are wound inward; draw both sides.
        )
        for g, v, f in parts
    ]
    boxes = server.scene.add_batched_meshes_simple(
        "/boxes",
        vertices=CUBE_VERTICES,
        faces=CUBE_FACES,
        batched_wxyzs=identity,
        batched_positions=offsets.astype(np.float32),
        batched_scales=np.full((lanes, 3), 1e-3, np.float32),
        batched_colors=box_colors,
        lod="off",
    )
    for r, (label, _) in enumerate(rows):
        server.scene.add_label(f"/labels/row_{r}", label, position=(-1.3, -r * ROW_SPACING, 1.2))
    column_labels = {}

    def label_columns(prior, ids):
        for handle in column_labels.values():
            handle.remove()
        column_labels.clear()
        for c, i in enumerate(ids):
            if i is None:
                continue
            ext = 100 * prior["size"][i].numpy()
            text = f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} cm"
            if prior["skills"] is not None:
                text = f"{prior['skills'][int(prior['class_ids'][i])]} · {text}"
            column_labels[c] = server.scene.add_label(
                f"/labels/column_{c}", text, position=(c * COLUMN_SPACING, -depth - 1.0, 0.3)
            )

    prior_menu = server.gui.add_dropdown("Prior", options=names, initial_value=names[0]) if len(names) > 1 else None

    def current():
        return priors[names[0] if prior_menu is None else prior_menu.value]

    skill = None
    present = sorted(
        {p["skills"][c] for p in priors.values() if p["skills"] is not None for c in set(p["class_ids"].tolist())},
        key=lambda n: next(p["skills"].index(n) for p in priors.values() if p["skills"] and n in p["skills"]),
    )
    if present:
        skill = server.gui.add_dropdown("Skill", options=("all", *present), initial_value="all")

    def order():
        """Window indices shown for the selected prior and skill, in file order."""
        prior = current()
        if skill is None or skill.value == "all" or prior["skills"] is None:
            return np.arange(len(prior["size"]))
        if skill.value not in prior["skills"]:
            return np.zeros(0, np.int64)
        return np.flatnonzero(prior["class_ids"] == prior["skills"].index(skill.value))

    page = server.gui.add_slider("Page", min=0, max=max(0, (len(order()) - 1) // cols), step=1, initial_value=0)

    def reset_page(_):
        page.max = max(0, (len(order()) - 1) // cols)
        page.value = 0

    for menu in (prior_menu, skill):
        if menu is not None:
            menu.on_update(reset_page)
    speed = server.gui.add_slider("Speed", min=0.1, max=1.0, step=0.1, initial_value=0.5)
    caption = server.gui.add_markdown("")
    server.gui.add_markdown(
        f"Each column is one box size in one color. Top row (darker shade): a real held-out {width}-frame window. "
        f"Rows below: {per_size} windows the prior generated for that box size, each from different noise. "
        "Boxes are tinted like their column and turn **black** while lifted with both hands farther than 8 cm. "
        "Each window loops in its own frame; samples are not reconstructions of the real window."
    )

    def camera(client):
        middle = np.array([(cols - 1) * COLUMN_SPACING / 2, -depth / 2, 0.5])
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = middle + np.array([-1.5, -0.75 * depth - 4.0, 4.0])
        client.camera.look_at = middle

    server.on_client_connect(camera)
    if args.share:
        print("Share URL:", server.request_share_url(), flush=True)
    print(f"Viewer ready: http://127.0.0.1:{server.get_port()}", flush=True)
    start = time.monotonic()
    drawn, shown_colors, shown_scales, labeled_page = None, None, None, None
    while True:
        time.sleep(1.0 / args.fps)
        if not server.get_clients():
            continue
        t = int((time.monotonic() - start) * 50 * speed.value) % width
        name = names[0] if prior_menu is None else prior_menu.value
        prior = priors[name]
        shown_skill = "all" if skill is None else skill.value
        key = (name, shown_skill, page.value, t)
        if key == drawn:
            continue
        drawn = key
        windows = order()
        ids = [int(windows[k]) if k < len(windows) else None for k in range(page.value * cols, (page.value + 1) * cols)]
        if labeled_page != (name, shown_skill, page.value):
            label_columns(prior, ids)
            labeled_page = (name, shown_skill, page.value)
        positions = np.full((lanes, len(parts), 3), -100.0)  # Lanes past the last window stay out of view.
        mats = np.tile(np.eye(3).reshape(1, 1, 9), (lanes, len(parts), 1))
        box_pos = np.full((lanes, 3), -100.0)
        box_wxyz = identity.astype(np.float64).copy()
        box_scale = np.full((lanes, 3), 1e-3)
        colors = box_colors.copy()
        off_now = 0
        for r, (_, sample) in enumerate(rows):
            for c, i in enumerate(ids):
                if i is None:
                    continue
                lane = r * cols + c
                p, m, center, rot, ext, off = pose(prior, sample, i, t)
                positions[lane], mats[lane] = p + offsets[lane], m
                box_pos[lane] = center + offsets[lane]
                box_wxyz[lane] = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
                box_scale[lane] = ext
                if off:
                    colors[lane] = HANDS_OFF_BOX
                    off_now += sample is not None
        quats = (
            Rotation.from_matrix(mats.reshape(-1, 3, 3))
            .as_quat()
            .reshape(lanes, len(parts), 4)[..., [3, 0, 1, 2]]
        )
        with server.atomic():
            for j, h in enumerate(robot):
                h.batched_positions = positions[:, j].astype(np.float32)
                h.batched_wxyzs = quats[:, j].astype(np.float32)
            boxes.batched_positions = box_pos.astype(np.float32)
            boxes.batched_wxyzs = box_wxyz.astype(np.float32)
            if shown_scales is None or not np.array_equal(shown_scales, box_scale):
                boxes.batched_scales = box_scale.astype(np.float32)
                shown_scales = box_scale
            if shown_colors is None or not np.array_equal(shown_colors, colors):
                boxes.batched_colors = colors
                shown_colors = colors
        shown = [i for i in ids if i is not None]
        first = page.value * cols
        head = prior["sampler"] if prior_menu is None else f"{name} ({prior['sampler']})"
        caption.content = (
            f"{head} · {'' if shown_skill == 'all' else shown_skill + ' '}windows {first + 1}-{first + len(shown)} "
            f"of {len(windows)} · frame {t + 1}/{width} · "
            f"generated windows with hands off the lifted box now: {off_now} of {len(shown) * per_size}"
        )


if __name__ == "__main__":
    main()
