import glob
from tensorboard.backend.event_processing import event_accumulator

def extract_eval_scalar_maxima(logdir_glob):
    for path in glob.glob(logdir_glob, recursive=True):
        if "tfevents" not in path:
            continue

        ea = event_accumulator.EventAccumulator(
            path,
            size_guidance={"scalars": 0}
        )
        ea.Reload()

        tags = [t for t in ea.Tags().get("scalars", []) if t.startswith("eval/")]
        if not tags:
            continue

        print(f"\nFILE: {path}")
        for tag in sorted(tags):
            pts = ea.Scalars(tag)
            if not pts:
                continue
            m = max(pts, key=lambda e: e.value).value
            short = tag.removeprefix("eval/")
            print(f"{short}: max={m:.4g}")

extract_eval_scalar_maxima("/Users/ddulaev/Desktop/YRAAA/**/events.out.tfevents.*")
