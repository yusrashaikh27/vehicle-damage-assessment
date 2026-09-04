# weights/

Put the trained checkpoint here as **`best.pt`**.

`core/detector.py` looks for exactly `weights/best.pt` (`DEFAULT_WEIGHTS`). Until that file
exists, `get_detector()` returns `StubDetector` and every page carries the placeholder
banner. That fallback is deliberate — see the note at the top of `core/detector.py`.

The file is produced by `notebooks/train_colab.ipynb` and mirrored to Google Drive at
`MyDrive/vda/best.pt` after every epoch. It is ~77.5 MB, which is why it is not committed:
see `.gitignore`. Google Drive will warn that it cannot virus-scan a file that size; that is
a size limit, not a detection.

The run that produced the current checkpoint: 100 epochs, 2.321 h, Colab T4, 31 Aug 2026.
Test-split figures and their caveats are in `docs/RESULTS.md`. Replacing this file changes
every number the app reports, so re-read that doc if you swap it.
