# Scientific Reports revision: presentation-only figures

The numerical results and experimental configurations in GitHub release `v1.0.0` are unchanged. The later `main`-branch files in this directory supply readability edits for the revised manuscript. Do not move or overwrite the published `v1.0.0` tag.

## Figure mapping

| Manuscript figure | Manuscript image path | Code |
| --- | --- | --- |
| Main Fig. 3 | `figure_2.png` | `figure_3_6_readability.py` |
| Main Fig. 5 | `figure_4.png` | `figure_typography_revision.py` |
| Main Fig. 6 | `figure_modal_relaxation.png` | `figure_3_6_readability.py` |
| Main Fig. 10 | `figure_mnist_training_audit.png` | `figure_typography_revision.py` |
| Supplementary Fig. S1 | `figure_5.png` | `figure_typography_revision.py` |

Figure 3, Figure 5, Figure 10, and S1 use the **original embedded PNGs** from the manuscript. The scripts change annotation-only regions; neither the graph data nor numerical results are modified. The original raster images and output files are supplied in the separately prepared figure bundle. Figure 6 uses the locked numerical input at `reported_results/modal_relaxation/audit.csv` to replot the same medians with enlarged axis typography, consistent colors, and a shared legend. Both PNG and vector PDF are generated for Figure 6.

## Reproduce the figure bundle

Run from the repository root with the `sources/` directory from the accompanying figure bundle:

```bash
python -m pip install -r requirements.txt
python -m pip install -r submission_figures/requirements.txt

python submission_figures/figure_typography_revision.py \
  --source-dir sources --output-dir figure_output

python submission_figures/figure_3_6_readability.py \
  --figure3-source sources/figure_3_original.png \
  --modal-csv reported_results/modal_relaxation/audit.csv \
  --output-dir figure_output
```

Without a clone of the repository, the final command can instead use the two median tables shipped in the bundle:

```bash
python submission_figures/figure_3_6_readability.py \
  --figure3-source sources/figure_3_original.png \
  --modal-summary-dir sources \
  --output-dir figure_output
```

To extract the unchanged Figure 3 image from the original 28-page main PDF instead of using the supplied PNG, pass `--main-pdf path/to/manuscript.pdf`. The PDF must have the original 2850-by-855-pixel Fig. 3 embedded on page 14.

The two scripts write SHA-256 provenance manifests. The later presentation changes do **not** alter the archived numerical tables, fixed seeds, solver settings, or simulation results. The published `v1.0.0` software DOI therefore continues to refer to its original release; if the final manuscript cites a newly archived software version, use that version's actual Zenodo DOI only after its publication.
