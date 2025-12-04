"""
ZGQ v8 Visualization Module
============================

IEEE-quality figure generation for research papers.

Usage:
    from visualization import generate_all_figures
    generate_all_figures(output_dir, format='pdf')

Or from command line:
    python -m visualization.generate_ieee_figures --output figures_ieee --format pdf
"""

from pathlib import Path

__all__ = ['generate_all_figures', 'COLORS', 'IEEE_SINGLE_COL', 'IEEE_DOUBLE_COL']

# Re-export main functionality
try:
    from .generate_ieee_figures import (
        generate_all_figures,
        COLORS,
        IEEE_SINGLE_COL,
        IEEE_DOUBLE_COL,
        IEEE_DPI,
        setup_ieee_style,
        fig_recall_vs_qps,
        fig_scaling_comparison,
        fig_latency_breakdown,
        fig_throughput_comparison,
        fig_recall_heatmap,
        fig_build_time_comparison,
        fig_pareto_frontier,
        fig_architecture_diagram,
        fig_recall_at_k,
        fig_memory_efficiency,
        fig_summary_radar,
    )
except ImportError:
    # Module not fully loaded yet
    pass
