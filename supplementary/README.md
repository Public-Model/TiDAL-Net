# Supplementary material of TiDAL-Net (TII-26-8813)

This folder contains the supplementary material for the manuscript "A Time-Drift-Aware Liquid Spatio-Temporal Graph Network for Irregular Telemetry Reconstruction of Low Earth Orbit Satellites". Due to space limitations, only the key content is analyzed and described in the manuscript. The remaining experimental descriptions, results, and formulas are collected here. The complete document is `Supplementary_Material.pdf` in the root of the repository.

## Files

| Path | Content |
|---|---|
| `../Supplementary_Material.pdf` | Complete supplementary document: Sections S1-S8, Figs. S1-S6, Tables S1-S13 |
| `equations.pdf` | Section S7 only: formulas of secondary steps moved from the method section |
| `figures/png/` | Figs. S1-S6, 600 dpi |
| `figures/pdf/`, `figures/svg/`, `figures/emf/` | Figs. S1-S6 as vector graphics (text converted to outlines in SVG and EMF) |
| `data/` | Values of Tables S1-S13 and of the curves in Figs. S1-S6 as CSV files |
| `source/` | LaTeX source of the supplementary document and the Python script that draws Figs. S1-S6 |

## Sections and related review comments

| Section | Topic | Figures and tables | Review comment |
|---|---|---|---|
| S1 | Experimental setup: datasets, degradation protocol and evaluated levels, baselines, training, metrics | Fig. S1, Tables S1-S3 | Reviewer 3, Comment 2 |
| S2 | Low affected-data ratios (0.5%-2.5%), error by position type, gated inference | Fig. S2, Tables S4-S6 | Reviewer 3, Comment 4 |
| S3 | Long consecutive gaps and the case without valid anchors | Fig. S3, Tables S7-S8 | Reviewer 3, Comment 5 |
| S4 | Reliability of the estimated uncertainty | Fig. S4, Tables S9-S10 | Reviewer 3, Comment 3 |
| S5 | Mission-mode shift and the training-derived candidate graph | Fig. S5, Tables S11-S12 | Reviewer 3, Comment 1 |
| S6 | Sensitivity of the graph-related parameters | Fig. S6, Table S13 | Reviewer 3, Comment 6 |
| S7 | Formulas of secondary steps moved from the manuscript | Eqs. (S2)-(S12) | - |
| S8 | Contents of this repository | - | - |

## Figure files

| Figure | File name |
|---|---|
| Fig. S1 | `FigS1_degradation_composition` |
| Fig. S2 | `FigS2_low_ratio` |
| Fig. S3 | `FigS3_long_gap` |
| Fig. S4 | `FigS4_uncertainty_reliability` |
| Fig. S5 | `FigS5_mission_mode_shift` |
| Fig. S6 | `FigS6_graph_parameter_sensitivity` |

## Rebuilding the figures and the document

```
cd source
python make_figures.py          # numpy, matplotlib; Inkscape >= 1.0 for the EMF files
pdflatex Supplementary_Material.tex
pdflatex Supplementary_Material.tex
```

MAE values are multiplied by 10^2 and averaged over 30 runs unless stated otherwise. Section, table, equation, and reference numbers without the prefix S refer to the revised manuscript.
