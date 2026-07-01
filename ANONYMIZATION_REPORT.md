# Anonymous Release Report

- Created: 2026-07-01T09:58:00
- Cleaned: 2026-07-01
- File count after cleanup: 56
- Total source size after cleanup: 0.91 MB
- Large files over 5 MB: 0
- Excluded-result pattern matches: 0
- Sensitive string matches: 0

Cleanup scope:

- Removed exploratory BVR/LRR/PRB code paths, Isaac visualization helpers,
  watch/continue scripts, local push instructions, obsolete experiment notes,
  plotting scripts, table-generation scripts, front-page figure scripts, and
  local launch wrappers.
- Kept the paper-relevant training, corrupted-belief recovery, non-learning
  baseline evaluation, robust-SAC recovery preparation, and terrain preprocessing
  code.
- Added the missing source-only `maps/` package required by the public
  reproducibility entry points.

No checkpoints, result tables, generated figures, local logs, raw DEM/DTM files,
or derived numpy terrain arrays are included in this anonymous release.
