# app/src/ml/analysis/__init__.py
"""
analysis plotting utilities
==========================
this package provides core utilities for visual analysis of ml build/train/tune/test stages
- common : shared functionalities
- analysis_build : standardized plotting utilities for building feature matrix stage
- analysis_train : standardized plotting utilities for train model stage
- analysis_val : standardized plotting utilities for validate model stage
- analysis_tune : standardized plotting utilities for tune model stage
- analysis_test : standardized plotting utilities for test model stage
"""

from .common import plot_latent_space
from .analysis_build import plot_log_candidates
from .analysis_train import plot_training_curves, plot_detailed_loss_curves, plot_detailed_mt_loss_curves
from .analysis_val import plot_error_histogram, plot_error_timeseries, summarize_validation, summarize_mt_validation, plot_regression_scatter, plot_attack_confusion_matrix, plot_attack_confidence_hist, plot_mt_anomaly_timeseries
from .analysis_tune import save_optuna_plots, plot_correlation_heatmap, plot_loss_curves_all_trials, plot_best_trial_learning_curve, plot_3d_scatter, plot_loss_component_analysis, plot_multi_loss_overview, plot_multi_weights_overview, plot_multi_weight_loss_correlation, plot_mt_loss_component_analysis, plot_attack_class_weights, plot_attack_class_balance, plot_multi_mt_weights_overview, plot_multi_mt_weight_loss_correlation, plot_multi_country_attack_weights
from .analysis_test import plot_error_curve, plot_intervals, plot_error_hist, plot_raw_with_errors, plot_true_pred_anomalies, plot_attack_timeline, plot_loss_components_timeseries 

__all__ = ["plot_latent_space", "plot_log_candidates", "plot_training_curves", "plot_detailed_loss_curves", "plot_detailed_mt_loss_curves", "plot_error_histogram", "plot_error_timeseries", "summarize_validation", "summarize_mt_validation", "plot_regression_scatter", "plot_attack_confusion_matrix", "plot_attack_confidence_hist", "plot_mt_anomaly_timeseries", "save_optuna_plots", "plot_correlation_heatmap", "plot_loss_curves_all_trials", "plot_best_trial_learning_curve", "plot_3d_scatter", "plot_loss_component_analysis", "plot_multi_loss_overview", "plot_multi_weights_overview", "plot_multi_weight_loss_correlation", "plot_mt_loss_component_analysis", "plot_attack_class_weights", "plot_attack_class_balance", "plot_multi_mt_weights_overview", "plot_multi_mt_weight_loss_correlation", "plot_multi_country_attack_weights", "plot_error_curve", "plot_intervals", "plot_error_hist", "plot_raw_with_errors", "plot_true_pred_anomalies", "plot_attack_timeline", "plot_loss_components_timeseries"]