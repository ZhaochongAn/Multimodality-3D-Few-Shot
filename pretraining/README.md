# Pretraining Stage

## Overview
This directory contains the backbone and IF head pretraining implementation for our MM-FSS framework. We build upon the [OpenScene](https://github.com/pengsongyou/openscene) codebase, extending it with supporting our backbone and few-shot setup.


## Usage
```bash
# Run pretraining
bash run/distill_strat.sh PATH_to_SAVE_BACKBONE config/scannet/ours_lseg_strat.yaml
```