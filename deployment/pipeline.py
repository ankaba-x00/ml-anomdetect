import time
import numpy as np
from datetime import datetime, timezone
from deployment.fetcher import run_fetcher
from deployment.inference import load_inference_bundle, run_inference

def detect_anomalies(
        country: str, 
        date_from: datetime, 
        date_to: datetime
    ):
    try: 
        bundle = load_inference_bundle(country)

        X_cont_df, X_cat_df, _, cat_dims2, _ = run_fetcher(country, date_from, date_to, bundle["scaler"])

        assert cat_dims2.keys() == bundle["cat_dims"].keys(), "[Error] Saved cat_dims fatal mismatch — rebuild features."

        X_cont = X_cont_df.values.astype(np.float32)
        X_cat = X_cat_df.values.astype(np.int64)
        ts = X_cont_df.index

        # --------------------
        # Run anomaly detection
        # --------------------
        results = run_inference(
            model=bundle["model"],
            X_cont=X_cont,
            X_cat=X_cat,
            threshold=bundle["threshold"],
            device=bundle["config"].device,
        )

        errors = results["errors"]
        threshold = results["threshold"]
        mask = results["mask"]
        starts = results["anomaly_starts"]
        ends = results["anomaly_ends"]

        # -------------------------
        # Print summary for CLI
        # -------------------------
        print(f"\n--- Result threshold {bundle['method']}---")
        print(f"Date = {date_from.date()}")
        print(f"Threshold = {threshold:.6f}")
        print(f"Detected {mask.sum()} anomalous samples")
        print(f"Detected {len(starts)} anomaly intervals")

        intervals = []
        for s, e in zip(starts, ends):
            interval_str = f"{ts[s]} - {ts[e]}  ({e-s} anomalies)"
            intervals.append(interval_str)
            print("  > Interval", interval_str)

        # -------------------------
        # Return summary for GUI
        # -------------------------
        return {
            "threshold": threshold,
            "num_anomalues": mask.sum(),
            "intervals" : intervals,
            "status": "ok"
        }
    except Exception as e:
        print(f"[ERROR] Inference failed internally: {e}")
        return {
            "status": "error"
        }
    except BaseException as e:
        print(f"[ERROR] Inference failed on system-level: {e}")
        return {
            "status": "error"
        }


if __name__=="__main__":
    import argparse, sys
    from datetime import datetime, timezone, timedelta
    from src.data.feature_engineering import COUNTRIES


    def _is_valid_date(date: str) -> bool:
        lower_bound = datetime(2025, 11, 15, tzinfo=timezone.utc)
        upper_bound = (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            - timedelta(days=1)
        )
        return lower_bound <= dt <= upper_bound


    parser = argparse.ArgumentParser(description="Run inference for country as CLI.")

    parser.add_argument(
        "-d", "--date",
        type=str, 
        default="11/14/2025",
        help="prediction date in format MM/DD/YYYT within range 11/15/2025 to yesterday [default: 11/15/2025]"
    )

    parser.add_argument(
        "target",
        type=str,
        help="<COUNTRY> e.g. 'US' for US model"
    )

    args = parser.parse_args()
    target, date = args.target, args.date
    if target not in COUNTRIES:
        print(f"No pre-trained model for {target}")
        print(f"Please choose from the following list: {COUNTRIES}")
        sys.exit(1)

    try:
        dt = datetime.strptime(date, "%m/%d/%Y").replace(tzinfo=timezone.utc)
        if dt and _is_valid_date(dt):
            DATE_FROM = DATE_TO = dt.replace(tzinfo=timezone.utc)
        else:
            raise ValueError
    except Exception:
        print(f"Date format not accepted: {date}.")
        print("Required format: MM/DD/YYYY")
        print("Required range: between 11/15/2025 and yesterday")
        sys.exit(1)

    detect_anomalies(
        args.target.upper(), 
        DATE_FROM, 
        DATE_TO
    )