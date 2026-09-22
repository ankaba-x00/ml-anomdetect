# app/src/data/analysis/__init__.py
"""
data analysis package 
=====================
this package provides core utilities for data analysis
- params : parameter dictionaries and configuration values
- time_utils : functions for time and timezone data conversions
- data_prep : utilities for transforming and enriching datasets
- analysis_fetch : standardized plotting utilities for fetch stage
"""
from .params import regions, continental_regions, timezones, tzoffset_to_regions
from .time_utils import timeit
from .data_prep import add_local_daytypes, add_local_daytimes, add_fluctuation_metrics, normalize_per_date, aggregate_directional, preprocess_matrix, evaluate_kmeans_over_k, fit_final_kmeans
from .analysis_fetch import boxplot_valdist, boxplot_valdist_daytypes, boxplot_valdist_daytimes, heatmap_activity_fluctuations, heatmap_log10_attack_profile, barplot_activity_fluctuations, barplot_top_attackers, heatmap_anomalies, lineplot_clustering_eval_curves, pca_clusterplot, radarchart_attack_fingerprint

__all__ = ["regions", "continental_regions", "timezones", "tzoffset_to_regions", "timeit", "add_local_daytypes", "add_local_daytimes", "add_fluctuation_metrics", "normalize_per_date", "aggregate_directional", "preprocess_matrix", "evaluate_kmeans_over_k", "fit_final_kmeans", "boxplot_valdist", "boxplot_valdist_daytypes", "boxplot_valdist_daytimes", "heatmap_activity_fluctuations", "heatmap_log10_attack_profile", "barplot_activity_fluctuations", "barplot_top_attackers", "heatmap_anomalies", "lineplot_clustering_eval_curves", "pca_clusterplot", "radarchart_attack_fingerprint"]