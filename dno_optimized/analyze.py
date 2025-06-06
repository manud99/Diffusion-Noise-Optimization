from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator
from flask import Flask, render_template, send_from_directory
import torch
import numpy as np
from scipy.ndimage import uniform_filter1d


def parse_args():
    parser = ArgumentParser(
        description="Show results from subfolders containing args.yml and TensorBoard logs."
    )
    parser.add_argument(
        "path", type=str, help="Path to the folder containing subfolders with results"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="Port to run the HTTP server on"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Flask's debug mode for auto-reloading",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate to filter results"
    )
    parser.add_argument(
        "--decorrelate_scale",
        type=float,
        default=None,
        help="Decorrelate scale to filter results",
    )
    return parser.parse_args()


def calculate_skating_ratio(motions):
    """Computes the foot skating ratio as described in the paper on page 6."""
    thresh_height = 0.05  # 5 cm above ground
    fps = 20.0
    thresh_vel = 0.50  # 20 cm /s # 2.5cm
    avg_window = 5  # frames

    # 10 left, 11 right foot. XZ plane, y up
    # shape of motions: [bs, 22, 3, max_len]
    verts_feet = (
        motions[:, [10, 11], :, :].detach().cpu().numpy()
    )  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = (
        np.linalg.norm(
            verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1], axis=2
        )
        * fps
    )  # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(
        verts_feet_plane_vel, axis=-1, size=avg_window, mode="constant", origin=0
    )  # [bs, 2, max_len-1]

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in adjacent frames
    feet_contact = np.logical_and(
        (verts_feet_height[:, :, :-1] < thresh_height),
        (verts_feet_height[:, :, 1:] < thresh_height),
    )  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel


def compute_jitter(
    predicted_position,
    fps=20,
):
    cal_jitter = (
        (
            (
                predicted_position[3:]
                - 3 * predicted_position[2:-1]
                + 3 * predicted_position[1:-2]
                - predicted_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return cal_jitter


def compute_metrics(
    # inital_motions,
    # generated_motions,
    # max_frames,
    # text,
    subfolder: Path,
    conf: object,
):

    results_file = np.load(subfolder / "results.npy", allow_pickle=True).item()
    all_motions = results_file["motion"]
    initial_motions = torch.tensor(all_motions[:1]).permute(0, 3, 1, 2)
    generated_motions = torch.tensor(all_motions[1:]).permute(0, 3, 1, 2)

    max_frames = min(conf.max_frames, int(conf.motion_length * conf.fps))
    text = conf.text_prompt
    metrics: dict = {
        "Foot skating": [],
        "Jitter": [],
        # "Content preservation": [],
        "Foot skating before": [],
        "Jitter before": [],
    }

    # See https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
    left_foot_id = 10
    right_foot_id = 11
    left_hand_id = 20
    right_hand_id = 21
    head_id = 15

    opt_batch_size = len(generated_motions) // len(initial_motions)
    bf_edit_content_list = []

    # Before edit
    for i in range(len(initial_motions)):
        before_edit_cut = initial_motions[i, :max_frames, :, :]
        skate_ratio, skate_vel = calculate_skating_ratio(
            before_edit_cut.permute(1, 2, 0).unsqueeze(0)
        )  # need input shape [bs, 22, 3, max_len]

        metrics["Foot skating before"].append(skate_ratio.item())
        metrics["Jitter before"].append(compute_jitter(before_edit_cut).item())

        if "jumping" in text or "jump" in text:
            before_edit_above_ground = (before_edit_cut[:, left_foot_id, 1] > 0.05) & (
                before_edit_cut[:, right_foot_id, 1] > 0.05
            )
            bf_edit_content_list.append(before_edit_above_ground)
        elif "raised hands" in text:
            before_edit_above_head = (
                before_edit_cut[:, left_hand_id, 1] > before_edit_cut[:, head_id, 1]
            ) & (before_edit_cut[:, right_hand_id, 1] > before_edit_cut[:, head_id, 1])
            bf_edit_content_list.append(before_edit_above_head)
        elif "crawling" in text:
            before_edit_head_below = before_edit_cut[:, head_id, 1] < 1.50
            bf_edit_content_list.append(before_edit_head_below)

    for i in range(len(generated_motions)):
        # Generated
        gen_cut = generated_motions[i, :max_frames, :, :]
        skate_ratio, skate_vel = calculate_skating_ratio(
            gen_cut.permute(1, 2, 0).unsqueeze(0)
        )

        metrics["Foot skating"].append(skate_ratio.item())
        metrics["Jitter"].append(compute_jitter(gen_cut).item())

        first_gen_idx = i // opt_batch_size
        # Compute content preservation
        if "jumping" in text or "jump" in text:
            # Compute the ratio of matched frames where the feet are above the ground or touching the ground
            # First compute which frames in the generated motion that the feet are above the ground
            gen_above_ground = (gen_cut[:, left_foot_id, 1] > 0.05) & (
                gen_cut[:, right_foot_id, 1] > 0.05
            )
            content_ratio = (
                gen_above_ground == bf_edit_content_list[first_gen_idx]
            ).sum() / max_frames
        elif "raised hands" in text:
            # Compute the ratio of matched frames where the hands are above the head
            gen_above_head = (gen_cut[:, left_hand_id, 1] > gen_cut[:, head_id, 1]) & (
                gen_cut[:, right_hand_id, 1] > gen_cut[:, head_id, 1]
            )
            content_ratio = (
                gen_above_head == bf_edit_content_list[first_gen_idx]
            ).sum() / max_frames
        elif "crawling" in text:
            # Compute the ratio of matched frames where the head is below 1.5m
            gen_head_below = gen_cut[:, head_id, 1] < 1.50
            content_ratio = (
                gen_head_below == bf_edit_content_list[first_gen_idx]
            ).sum() / max_frames
        else:
            content_ratio = torch.tensor(0.0, device=gen_cut.device)

        # metrics["Content preservation"].append(content_ratio.item())

    return metrics


def get_results(
    path: Path, lr: float | None = None, decorrelate_scale: float | None = None
):
    subfolders = [
        f
        for f in path.iterdir()
        if f.is_dir() and (f / "args.yml").is_file() and (f / "logs").is_dir()
    ]
    if not subfolders:
        raise ValueError(
            f"No valid subfolders found in '{path}' containing 'args.yml' and 'logs' directory."
        )

    renamed_prompts = {
        "person is walking casually with their arms slightly swinging": "a person is walking casually with their arms slightly swinging",
        "he slowly walking forward towards something": "a person slowly walking forward towards something",
        "person walking with their arms swinging back to front and walking in a general circle": "a person walking with their arms swinging back to front and walking in a general circle",
    }
    results = []
    for subfolder in sorted(subfolders):
        args_file = subfolder / "args.yml"
        logs_dir = subfolder / "logs"
        video_file = subfolder / "samples_00_to_03.mp4"

        # Read and parse args.yml file
        conf = OmegaConf.create(args_file.read_text())
        args = {
            "optimizer": conf.dno.optimizer,
            "lr": conf.dno.lr,
            "decorrelate_scale": conf.dno.decorrelate_scale,
            "diff_penalty_scale": conf.dno.diff_penalty_scale,
            # 'damping_mode': conf.dno.levenbergMarquardt.damping_strategy.damping_mode,
        }

        if lr is not None and args["lr"] != lr:
            continue
        if (
            decorrelate_scale is not None
            and args["decorrelate_scale"] != decorrelate_scale
        ):
            continue

        # Load TensorBoard event data
        ea = event_accumulator.EventAccumulator(str(logs_dir))
        ea.Reload()
        tags = ea.Tags()["scalars"]

        interesting_tags = [
            ("01_loss/trial_0", "Total Loss"),
            ("02_loss_objective/trial_0", "Objective Loss"),
            ("03_loss_diff/trial_0", "Diff Loss"),
            ("04_loss_decorrelate/trial_0", "Decorrelate Loss"),
        ]
        losses = {}
        for tag, label in interesting_tags:
            if tag not in tags:
                continue
            scalars = ea.Scalars(tag)
            if not scalars:
                continue
            # extract the smallest non-nan value from the scalars
            min_value = min(
                s.value
                for s in scalars
                if s.value is not None
                and not isinstance(s.value, float)
                or not s.value != s.value
            )  # Exclude NaN values
            losses[label] = min_value

        metrics = compute_metrics(subfolder, conf)

        # Prepare result entry
        result_entry = {
            "text_prompt": (
                renamed_prompts[conf.text_prompt]
                if conf.text_prompt in renamed_prompts
                else conf.text_prompt
            ),
            "name": subfolder.name,
            "args": args,
            "losses": losses,
            "metrics": metrics,
            "video": (
                str(video_file.relative_to(path)) if video_file.is_file() else None
            ),
        }
        results.append(result_entry)

    return sorted(results, key=lambda x: x["text_prompt"])


def create_app(results, base_path):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template(
            "video_template.html", results=results, base_path=base_path
        )

    @app.route("/results")
    def results_page():
        return render_template(
            "results_template.html", results=results, base_path=base_path
        )

    @app.route("/video/<path:filename>")
    def video(filename):
        # Serve video files from the results directory
        return send_from_directory(base_path, filename)

    return app


def main():
    args = parse_args()
    path = Path(args.path)
    results = get_results(path, args.lr, args.decorrelate_scale)
    app = create_app(results, str(path))
    print(f"Results will be served on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.reload)


if __name__ == "__main__":
    main()
