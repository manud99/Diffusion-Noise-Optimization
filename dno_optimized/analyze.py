from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator
from flask import Flask, render_template, send_from_directory

# Create an http server to show the parameters, losses, and videos all on one HTML page
# Parameters come from `args.yml` file in subfolders
# Losses come from TensorBoard event files `logs/events.out.tfevents...` in subfolders
# Videos come from video file `samples_00_to_03.mp4` in subfolders

def parse_args():
    parser = ArgumentParser(description="Show results from subfolders containing args.yml and TensorBoard logs.")
    parser.add_argument("path", type=str, help="Path to the folder containing subfolders with results")
    parser.add_argument('-p', '--port', type=int, default=8000, help="Port to run the HTTP server on")
    parser.add_argument('--reload', action='store_true', help="Enable Flask's debug mode for auto-reloading")
    return parser.parse_args()

def get_results(path: Path):
    subfolders = [f for f in path.iterdir() if f.is_dir() and (f / "args.yml").is_file() and (f / "logs").is_dir()]
    if not subfolders:
        raise ValueError(f"No valid subfolders found in '{path}' containing 'args.yml' and 'logs' directory.")
    
    results = []
    for subfolder in sorted(subfolders):
        args_file = subfolder / "args.yml"
        logs_dir = subfolder / "logs"
        video_file = subfolder / "samples_00_to_03.mp4"
        
        # Read and parse args.yml file
        conf = OmegaConf.create(args_file.read_text())
        args = {
            'lr': conf.dno.lr,
            'decorrelate_scale': conf.dno.decorrelate_scale,
            'damping_mode': conf.dno.levenbergMarquardt.damping_strategy.damping_mode
        }

        # Load TensorBoard event data
        ea = event_accumulator.EventAccumulator(str(logs_dir))
        ea.Reload()
        tags = ea.Tags()['scalars']
        interesting_tags = ['01_loss/trial_0', '02_loss_objective/trial_0', '03_loss_diff/trial_0', '04_loss_decorrelate/trial_0']
        losses = {}
        for tag in interesting_tags:
            if tag not in tags:
                continue
            scalars = ea.Scalars(tag)
            if not scalars:
                continue
            # extract the smallest non-nan value from the scalars
            min_value = min(s.value for s in scalars if s.value is not None and not isinstance(s.value, float) or not s.value != s.value)  # Exclude NaN values
            losses[tag] = min_value

        # Prepare result entry
        result_entry = {
            'name': subfolder.name,
            'args': args,
            'losses': losses,
            'video': str(video_file.relative_to(path)) if video_file.is_file() else None
        }
        results.append(result_entry)
    
    return results

def create_app(results, base_path):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("results_template.html", results=results)

    @app.route("/video/<path:filename>")
    def video(filename):
        # Serve video files from the results directory
        file = Path(base_path) / filename
        print(f"Serving video: {base_path}/{filename} -> {file.is_file()}")

        return send_from_directory(base_path, filename)

    return app

def main():
    args = parse_args()
    path = Path(args.path)
    results = get_results(path)
    app = create_app(results, str(path.resolve()))
    print(f"Results will be served on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.reload)

if __name__ == "__main__":
    main()
