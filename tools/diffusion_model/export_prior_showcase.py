"""Export a skill-conditioned prior's generated windows as one HTML page of offline Viser replays.

One tab per skill, laid out like view_prior_samples.py: a column and a color per box size,
the real held-out window on top (in a darker shade) and below it the windows the prior
generated for that skill and box size, one row per sample, each from its own noise. Columns are real windows whose
box travels at least --min_travel_m (carry: --min_travel_carry_m), spread from the smallest
to the largest box; the generated windows are shown as drawn, never filtered. Each window
loops at --speed. The page embeds one copy of Viser's client and every scene, so it opens
straight from disk. Viser's offline player does not draw labels, so row and column names
are text images lying on the ground.

    source scripts/source_hiphi_retargeting.sh   # in FAR-Holosoma: viser + mujoco
    PYTHONPATH=/path/to/MimicKit:$PYTHONPATH "$FAR_RETARGET_PYTHON" \\
        tools/diffusion_model/export_prior_showcase.py --samples <run>/eval/samples_validation_final_ddpm50_x5.npz \\
        --scene <any replay scene.xml with G1 meshes> --notes notes.html --output showcase.html
"""

import argparse
import base64
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from view_prior_samples import (  # noqa: E402
    COLUMN_COLORS,
    COLUMN_SPACING,
    CUBE_FACES,
    CUBE_VERTICES,
    HANDS_OFF_BOX,
    MAX_SAMPLE_ROWS,
    POSE_KEYS,
    ROW_SPACING,
    box_distance,
    lane_colors,
    robot_parts,
    shade,
)

from mimickit_smp.dataset import CarryWindowDataset  # noqa: E402
from mimickit_smp.decode import decode_window, matrix_to_wxyz  # noqa: E402
from mimickit_smp.features import FeatureSchema, encode_window  # noqa: E402

BOX_POS = slice(201, 204)  # Box center in the window frame (feature schema v3).
STAGE_NAMES = {
    "carry": ("pickup", "lifted", "set-down"),
    "push": ("approach", "box moving", "settle"),
    "slide": ("approach", "box moving", "settle"),
}


def floor_text(server, name, text, x, y, height=0.34, px=72):
    """Dark text on a white card lying on the ground, readable from the default camera (front, above)."""
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", px)
    except OSError:
        font = ImageFont.load_default(size=px)
    card = Image.new("RGB", (int(font.getlength(text)) + px // 2, int(px * 1.35)), (255, 255, 255))
    ImageDraw.Draw(card).text((px // 4, px // 10), text, fill=(15, 15, 15), font=font)
    image = np.flipud(np.asarray(card))  # Image rows run toward the camera on the ground plane.
    server.scene.add_image(
        name, image, render_width=height * image.shape[1] / image.shape[0], render_height=height,
        format="png", cast_shadow=False, receive_shadow=False, position=(x, y, 0.005),
    )


def pick_columns(real, size, class_ids, skill_id, columns, min_travel):
    """Real windows of one skill whose box travels >= min_travel, spread from small to large boxes."""
    ids = np.flatnonzero(class_ids == skill_id)
    travel = np.linalg.norm(real[ids, -1, BOX_POS] - real[ids, 0, BOX_POS], axis=-1)
    pool = ids[travel >= min_travel]
    if len(pool) < columns:
        pool = ids[np.argsort(-travel, kind="stable")[:columns]]
    pool = pool[np.argsort(np.prod(size[pool], axis=-1), kind="stable")]
    return np.unique(pool[np.linspace(0, len(pool) - 1, min(columns, len(pool))).round().astype(int)])


def window_sources(samples, schema):
    """Trial, actor and start of each drawn real window, by replaying sample_carry_prior.py's draw.

    Returns None unless the replayed windows encode exactly to the saved real windows.
    """
    model_dir = Path(str(samples["model"])).parent
    config = yaml.safe_load((model_dir / "diffusion_config.yaml").read_text())
    dataset = CarryWindowDataset(
        str(samples["index"]), split=str(samples["split"]), quality_checks=config.get("quality_checks") or ()
    )
    present = [c for c in sorted(dataset.class_names) if (dataset.window_class_ids == c).any()]
    rng = np.random.default_rng(0)  # sample_carry_prior.py's default seed.
    per = 2048 // len(present)
    pick = np.concatenate(
        [
            rng.choice(np.flatnonzero(dataset.window_class_ids == c), min(per, int((dataset.window_class_ids == c).sum())), replace=False)
            for c in present
        ]
    )
    if len(pick) != len(samples["real_motion"]):
        return None

    def source(i):
        window = dataset[int(pick[i])]
        motion, _ = encode_window(torch.as_tensor(window["states"]).double()[None], schema)
        if not np.allclose(motion[0].float().numpy(), samples["real_motion"][i], atol=1e-4):
            raise ValueError("Replayed window differs from the saved one")
        segment = dataset.segments[int(dataset.window_segments[pick[i]])]
        return {"trial": segment["trial_id"], "actor": segment["actor_id"], "start_state": int(dataset.window_starts[pick[i]])}

    return source


def export_scene(skill, ids, samples, schema, model, parts, per_size, speed):
    """Record one window loop of the chosen columns into a .viser payload."""
    size = torch.from_numpy(samples["box_size"][ids]).double()

    def decode(motion):
        d = decode_window(torch.from_numpy(motion).double(), size, schema)
        return {k: d[k].float() for k in POSE_KEYS}

    rows = [("real", decode(samples["real_motion"][ids]))] + [
        (f"sample {k + 1}", decode(samples["generated_motion"][ids, k])) for k in range(per_size)
    ]
    data = mujoco.MjData(model)
    geom_ids = np.asarray([p[0] for p in parts])
    hands = [model.body(n).id for n in ("left_rubber_hand_link", "right_rubber_hand_link")]
    cols, width = len(ids), schema.window_frames
    lanes = len(rows) * cols
    offsets = np.array([[c * COLUMN_SPACING, -r * ROW_SPACING, 0.0] for r in range(len(rows)) for c in range(cols)])
    robot_colors, box_colors = lane_colors([("real", None)] + [(label, k) for k, (label, _) in enumerate(rows[1:])], cols)

    def frame(t):
        positions = np.zeros((lanes, len(parts), 3))
        quats = np.zeros((lanes, len(parts), 4))
        box_pos, box_wxyz, box_scale = np.zeros((lanes, 3)), np.zeros((lanes, 4)), np.zeros((lanes, 3))
        colors = box_colors.copy()
        for r, (_, d) in enumerate(rows):
            for c in range(cols):
                lane = r * cols + c
                data.qpos[:3] = d["root_pos"][c, t].numpy()
                data.qpos[3:7] = matrix_to_wxyz(d["root_rot"][c, t]).numpy()
                data.qpos[7:36] = d["joint_pos"][c, t].numpy()
                data.qpos[36:39] = [0.0, 0.0, -10.0]  # Hide the scene's own box.
                data.qpos[39:43] = [1.0, 0.0, 0.0, 0.0]
                mujoco.mj_kinematics(model, data)
                positions[lane] = data.geom_xpos[geom_ids] + offsets[lane]
                quats[lane] = Rotation.from_matrix(data.geom_xmat[geom_ids].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
                center, rot = d["box_pos"][c, t].numpy(), d["box_rot"][c, t].numpy()
                ext = d["box_extents"][c].numpy()
                box_pos[lane], box_scale[lane] = center + offsets[lane], ext
                box_wxyz[lane] = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
                bottom = center[2] - 0.5 * np.abs(rot[2]) @ ext
                if bottom > 0.10 and min(box_distance(data.xpos[h], center, rot, ext) for h in hands) > 0.08:
                    colors[lane] = HANDS_OFF_BOX
        return positions.astype(np.float32), quats.astype(np.float32), box_pos, box_wxyz, box_scale, colors

    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        server.scene.set_up_direction("+z")
        depth = (len(rows) - 1) * ROW_SPACING
        server.scene.add_grid(
            "/ground",
            width=cols * COLUMN_SPACING + 5,
            height=depth + 4,
            cell_size=0.5,
            position=((cols - 1) * COLUMN_SPACING / 2 - 0.5, -depth / 2, 0.0),
        )
        positions, quats, box_pos, box_wxyz, box_scale, colors = frame(0)
        robot = [
            server.scene.add_batched_meshes_simple(
                f"/robot/{g}",
                vertices=v,
                faces=f,
                batched_wxyzs=quats[:, j],
                batched_positions=positions[:, j],
                batched_colors=robot_colors,
                side="double",  # A few source triangles are wound inward; draw both sides.
            )
            for j, (g, v, f) in enumerate(parts)
        ]
        boxes = server.scene.add_batched_meshes_simple(
            "/boxes",
            vertices=CUBE_VERTICES,
            faces=CUBE_FACES,
            batched_wxyzs=box_wxyz.astype(np.float32),
            batched_positions=box_pos.astype(np.float32),
            batched_scales=box_scale.astype(np.float32),
            batched_colors=colors,
            lod="off",
        )
        for r, (label, _) in enumerate(rows):
            floor_text(server, f"/labels/row_{r}", label, -1.75, -r * ROW_SPACING)
        for c, i in enumerate(ids):
            ext = 100 * samples["box_size"][i]
            floor_text(server, f"/labels/column_{c}", f"{ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f}", c * COLUMN_SPACING, -depth - 1.1)
        middle = np.array([(cols - 1) * COLUMN_SPACING / 2, -depth / 2, 0.5])
        server.initial_camera.position = tuple(middle + np.array([-1.0, -0.6 * depth - 3.5, 3.6]))
        server.initial_camera.look_at = tuple(middle)
        server.initial_camera.up = (0.0, 0.0, 1.0)
        recording = server.get_scene_serializer()
        shown_colors = colors
        for t in range(width):
            if t:
                recording.insert_sleep(1.0 / (50.0 * speed))
            positions, quats, box_pos, box_wxyz, _, colors = frame(t)
            for j, h in enumerate(robot):
                h.batched_positions = positions[:, j]
                h.batched_wxyzs = quats[:, j]
            boxes.batched_positions = box_pos.astype(np.float32)
            boxes.batched_wxyzs = box_wxyz.astype(np.float32)
            if t == 0 or not np.array_equal(colors, shown_colors):
                boxes.batched_colors = colors
                shown_colors = colors
        recording.insert_sleep(1.0 / (50.0 * speed))  # Hold the last state for one frame before the loop.
        return recording.serialize()
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True, help="sample_carry_prior.py output")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--notes", type=Path, help="HTML fragment placed above the replays")
    parser.add_argument("--title", default="Skill-conditioned SMP prior")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--min_travel_m", type=float, default=0.10)
    parser.add_argument("--min_travel_carry_m", type=float, default=0.20)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = dict(np.load(args.samples, allow_pickle=False))
    schema = FeatureSchema.from_dict(json.loads(Path(str(samples["index"])).read_text())["feature_schema"]).current_features()
    if schema.sha256 != str(samples["feature_schema_sha256"]):
        raise ValueError("Samples were made with a different feature schema")
    if samples["generated_motion"].ndim != 4:
        raise ValueError("Expected several samples per box size")
    per_size = min(samples["generated_motion"].shape[1], MAX_SAMPLE_ROWS)
    skills = samples["class_names"].tolist()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    parts = robot_parts(model)
    try:
        source = window_sources(samples, schema)
    except ValueError as error:
        print(f"No window provenance: {error}", flush=True)
        source = None
    scenes = []
    for skill_id, skill in enumerate(skills):
        if not (samples["class_ids"] == skill_id).any():
            continue
        travel = args.min_travel_carry_m if skill == "carry" else args.min_travel_m
        ids = pick_columns(samples["real_motion"], samples["box_size"], samples["class_ids"], skill_id, args.columns, travel)
        payload = export_scene(skill, ids, samples, schema, model, parts, per_size, args.speed)
        names = STAGE_NAMES.get(skill, ("pickup", "lifted", "set-down"))
        columns = []
        for c, i in enumerate(ids):
            ext = 100 * samples["box_size"][i]
            moved = 100 * np.linalg.norm(samples["real_motion"][i, -1, BOX_POS] - samples["real_motion"][i, 0, BOX_POS])
            where = f"{names[int(samples['stage_ids'][i])]}, box moves {moved:.0f} cm"
            if source is not None:
                s = source(int(i))
                where += f", {s['actor']} {s['trial']}"
            swatch = f'<span class="swatch" style="background:rgb{COLUMN_COLORS[c % len(COLUMN_COLORS)][1]}"></span>'
            columns.append(f"{swatch}{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} cm ({html.escape(where)})")
        caption = (
            f"Columns, left to right: {'; '.join(columns)}. Real windows from validation actors; "
            f"{per_size} generated windows per box size ({html.escape(str(samples['sampler']))}); playback at {args.speed:g}x."
        )
        scenes.append({"id": skill, "title": skill.capitalize(), "caption": caption, "scene": base64.b64encode(payload).decode()})
        print(f"{skill}: {len(ids)} columns, scene {len(payload) / 2**20:.1f} MiB", flush=True)
    base = COLUMN_COLORS[0][1]
    legend = (
        "Each color is one box size (one column; sizes and colors are listed under each replay). "
        f'<span><span class="swatch" style="background:rgb{shade(base)}"></span>darker shade: the real held-out window (top row)</span>'
        f'<span><span class="swatch" style="background:rgb{base}"></span>full color: the {per_size} generated windows below it</span>'
        f'<span><span class="swatch" style="background:rgb{HANDS_OFF_BOX}"></span>box lifted with both hands &gt; 8 cm away</span>'
    )
    client = (Path(viser.__file__).parent / "client/build/index.html").read_text()
    packed = json.dumps({"client": base64.b64encode(client.encode()).decode(), "scenes": scenes}).replace("<", "\\u003c")
    page = Path(__file__).with_name("prior_showcase.html").read_text()
    for key, value in (
        ("__TITLE__", html.escape(args.title)),
        ("__GENERATED_UTC__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("__NOTES__", args.notes.read_text() if args.notes else ""),
        ("__LEGEND__", legend),
        ("__DATA__", packed),
    ):
        page = page.replace(key, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
