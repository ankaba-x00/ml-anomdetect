#!/usr/bin/env python3
"""
Run interference pipeline as CLI for specified date 
A) AE/VAE/MTAE to predict traffic anomalies
B) MTAE to predict L3/L7 intensities and attack types
- loads model bundle
- fetches new data for specified date
- applies model
- prints summary 
- (optional) returns results to api

Usage:
    python -m app.deployment.use_mdoel [-d <date>] <MODEL> <COUNTRY>
"""

from datetime import datetime, timezone
import numpy as np
from pathlib import Path
from typing import Any

from app.src.data import ID_TO_ATTACK, ISO_3166_alpha2
from app.deployment.features import build_features
from app.deployment.fetcher import run_fetch
from app.deployment.loader import load_model_bundle
from app.src.ml.training.anomaly_utils import get_threshold
from app.src.ml.training.evaluate_ae import apply_model
from app.src.ml.training.evaluate_mt import apply_mt_model


FILE_DIR = Path(__file__).resolve().parent
MODELS_DIR = FILE_DIR / "models"


def use_model(
    ae_type: str,
    country: str, 
    date_from: datetime, 
    date_to: datetime
) -> dict[str, Any]:
    "Inference pipeline to fetch new CLoudflare data, generate feature matrix and apply model to make predictions."

    try: 
        bundle = load_model_bundle(ae_type, country)

        newdata = run_fetch(country, date_from, date_to)

        if ae_type in ["ae", "vae"]:
            X_cont, X_cat, num_cont, cat_dims, = build_features(country, newdata)
        else:
            X_cont, X_cat, y_l3, y_l7, y_at, num_cont, cat_dims = build_features(country, newdata, supervised=True)
        
        assert bundle["model_num_cont"] == num_cont, "[ERROR] num_cont mismatch between scaler and feature matrix"
        assert bundle["model_cat_dims"].keys() == cat_dims.keys(), "[ERROR] cant_dims mismatch between scaler and feature matrix"

        Xc = X_cont.values.astype(np.float32)
        Xk = X_cat.values.astype(np.int64)
        ts = X_cont.index
        if ae_type in ["mtae"]:
            y3 = y_l3.values.astype(np.float32)
            y7 = y_l7.values.astype(np.float32)
            ya = y_at.values.astype(np.int64)

        Xc_scaled = bundle["scaler"].transform(Xc).astype(np.float32)

        # --------------------
        # Apply model
        # --------------------
        if ae_type in ["ae", "vae"]:
            result = apply_model(
                model=bundle["model"],
                X_cont=Xc_scaled,
                X_cat=Xk,
                loss_weights=bundle["loss_weights"],
                device=bundle["config"].device,
                temperature=bundle.get("temperature", 1.0),
                beta=getattr(bundle["config"], "beta", 1.0),
                threshold=bundle["threshold"]
            )
        else:
            # --------------------
            # Set quantiles
            # --------------------
            med_q = bundle["config"].quantiles.index(0.5)
            pred_quantiles = bundle["config"].quantiles[med_q:]
            print(f"[INFO] Quantiles used for prediction: {pred_quantiles}")

            result = apply_mt_model(
                model=bundle["model"],
                X_cont=Xc_scaled,
                X_cat=Xk,
                y_l3=y3,
                y_l7=y7,
                y_attack=ya,
                loss_weights=bundle["loss_weights"],
                attack_type_weights=bundle["attack_type_weights"],
                pred_quantiles=pred_quantiles,
                method=bundle["method"],
                min_length=1,
                merge_gap=0,
                device=bundle["config"].device
            )

        scores = result["scores"]
        threshold = result["threshold"]
        mask = result["mask"]
        starts = result["anomaly_starts"]
        ends = result["anomaly_ends"]

        if ae_type in ["mtae"]:
            l3_pred = result["l3_pred"][0.5]
            l7_pred = result["l7_pred"][0.5]
            at_pred = result["at_pred"]

        # -------------------------
        # Print summary
        # -------------------------
        print("\n--- Inference Summary ---")
        print(f"Date = {date_from.date()}")
        print(f"Total samples = {len(scores)}")
        print(f"Threshold = {threshold:.6f}")
        print(f"Flagged samples = {mask.sum()}")
        print(f"Flagged intervals = {len(starts)}\n")

        intervals, l3_int, l7_int, att_type = [], [], [], []
        for s, e in zip(starts, ends):
            interval_str = f"{ts[s].strftime('%d/%m/%y,%H:%M')} - {ts[e-1].strftime('%d/%m/%y,%H:%M')} ({e-s} anomalies)"
            intervals.append(interval_str)
            print(f"  > Interval {interval_str}")
            if ae_type in ["mtae"]:
                l3 = l3_pred[s:e].tolist()
                l7 = l7_pred[s:e].tolist()
                at = [ID_TO_ATTACK[t] for t in at_pred[s:e]]
                l3_int.append(l3)
                l7_int.append(l7)
                att_type.append(at)
                print(f"\tl3: {l3}\n\tl7: {l7}\n\ttype: {at}")
        
        print(f"\nScore Statistics:")
        print(f"Min:       {scores.min():.6f}")
        print(f"Max:       {scores.max():.6f}")
        print(f"Mean:      {scores.mean():.6f}")
        print(f"Std:       {scores.std():.6f}")
        print(f"Score {bundle['method']}: {get_threshold(bundle['method'], scores):4f}\n")        

        # -------------------------
        # Return summary
        # -------------------------
        summary = {
            "country": country,
            "threshold": threshold,
            "num_anomalies": mask.sum(),
            "intervals" : intervals,
            "status": "ok"
        }
        if ae_type in ["mtae"]:
            summary |= {
                "l3_intensity": l3_int,
                "l7_intensity": l7_int,
                "attack_type": att_type
            }
        return summary
    except Exception as e:
            print(f"[ERROR] Inference failed internally: {e}")
            summary = {
                "country": country,
                "threshold": None,
                "num_anomalies": 0,
                "intervals": [],
                "status": f"error: {e}"
            }
            if ae_type in ["mtae"]:
                summary |= {
                    "l3_intensity": [],
                    "l7_intensity": [],
                    "attack_type": []
                }
            return summary


if __name__=="__main__":
    import argparse, sys
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    from app.src.data.feature_engineering import COUNTRIES


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
        default="11/15/2025",
        help="prediction date MM/DD/YYYY within range 11/15/2025 to yesterday [default: 11/15/2025]"
    )

    parser.add_argument(
        "model",
        help="<ae|vae|mtae> model to train"
    )

    parser.add_argument(
        "target",
        type=str,
        help="<COUNTRY> e.g. 'US'"
    )

    args = parser.parse_args()

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae", "mtae"]:
        parser.print_help()
        exit(1)
    
    target, date = args.target.upper(), args.date

    if target not in ISO_3166_alpha2:
        parser.print_help()
        print(f"[ERROR] Invalid target: {target}")
        exit(1)

    MODELS_DIR = Path(__file__).resolve().parent / "models"
    model_path = MODELS_DIR / f"{ae_type.upper()}" / f"{target.upper()}_autoencoder.pt"
    if not model_path.exists():
        print(f"[ERROR] No pre-trained model for {target.upper()}")
        sys.exit(1)
    
    try:
        dt = datetime.strptime(date, "%m/%d/%Y").replace(tzinfo=timezone.utc)
        if dt and _is_valid_date(dt):
            DATE_FROM = DATE_TO = dt.replace(tzinfo=timezone.utc)
        else:
            raise ValueError
    except Exception:
        print(f"Date format not accepted: {date}")
        print("Required format: MM/DD/YYYY")
        print("Required range: between 11/15/2025 and yesterday")
        sys.exit(1)

    use_model(
        ae_type,
        target,
        DATE_FROM,
        DATE_TO
    )