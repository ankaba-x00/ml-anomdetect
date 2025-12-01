import time
import numpy as np
from datetime import datetime, timezone
from deployment.fetcher import run_fetcher
from deployment.inference import load_inference_bundle, run_inference

def live_loop(
        country: str, 
        date_from: datetime, 
        date_to: datetime
    ):
    
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
    # Print summary
    # -------------------------
    print(f"\n--- Result threshold {bundle['method']}---")
    print(f"Threshold = {threshold:.6f}")
    print(f"Detected {mask.sum()} anomalous samples")
    print(f"Detected {len(starts)} anomaly intervals\n")

    for s, e in zip(starts, ends):
        print(f"  > Interval {ts[s]} - {ts[e]}  ({e-s} anomalies)")



if __name__=="__main__":

    country = "TW"
    DATE_FROM = datetime(2025, 11, 14, tzinfo=timezone.utc)
    DATE_TO   = datetime(2025, 11, 14, tzinfo=timezone.utc)
    live_loop(country, DATE_FROM, DATE_TO)
