# External references (datasets)

This directory contains external, third-party and/or machine-local datasets used by experiments.
Only small metadata and bootstrap scripts are committed; the dataset contents themselves are kept out of git.

Managed references:

- `coco2017/` — MS COCO 2017 images + annotations (local external storage)
  - `source-data`: expected to contain `train2017/`, `val2017/`, `test2017/`, and `annotations/`
  - Bootstrap: `bash datasets/coco2017/bootstrap.sh`

Populate/repair all dataset references:

```bash
bash datasets/bootstrap.sh
```

