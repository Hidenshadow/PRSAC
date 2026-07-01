# Anonymous Release Report

- Created: 2026-07-01T09:58:00
- Cleaned: 2026-07-01
- File count after cleanup: 99
- Total source size after cleanup: 1.31 MB
- Large files over 5 MB: 0
- Excluded-result pattern matches: 0
- Sensitive string matches: 0

Cleanup scope:

- Removed exploratory BVR/LRR/PRB code paths, Isaac visualization helpers,
  watch/continue scripts, local push instructions, and obsolete experiment notes.
- Kept the paper-relevant training, corrupted-belief recovery, non-learning
  baseline evaluation, terrain preprocessing, table, and figure-generation code.
- Added the missing source-only `maps/` package required by the public
  reproducibility entry points.

No checkpoints, result tables, generated figures, local logs, raw DEM/DTM files,
or derived numpy terrain arrays are included in this anonymous release.
