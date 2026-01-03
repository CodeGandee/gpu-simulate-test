# COCO 2017 (external reference)

This folder manages a machine-local reference to the MS COCO 2017 dataset directory.

## Layout

- `source-data` (symlink, ignored by git): points at the local COCO root directory that contains:
  - `train2017/`, `val2017/`, `test2017/`
  - `annotations/` (e.g. `instances_train2017.json`, `captions_val2017.json`)
- `bootstrap.sh`: recreates/repairs the `source-data` symlink.

## Bootstrap

```bash
# Option A (recommended): point GSIM_DATASETS_ROOT at your dataset storage root
export GSIM_DATASETS_ROOT=/path/to/datasets
bash datasets/coco2017/bootstrap.sh

# Option B: rely on the repository's detected default path (if it exists on this machine)
bash datasets/coco2017/bootstrap.sh
```
