# Run a lengthy experiment (tmux + persistent logs)

Use this pattern for long-running sim/real experiments so you can detach safely and still monitor progress from another shell.

## 1) Always run in `tmux`

Create (or attach to) a dedicated session:

- Create: `tmux new -s exp`
- Attach later: `tmux attach -t exp`
- Detach (leave it running): press `Ctrl-b` then `d`
- List sessions: `tmux ls`

Tip: use one session per experiment family (or per run) to keep history clean.

## 2) Always write a live log file under the output directory

Rationale: experiments can take a long time; a live log file lets you check progress without opening `tmux`.

### Recommended convention

In the run output directory (e.g. a `results/reports/...` or `tmp/...` run dir), create:

- `run.log`: full stdout/stderr of the run (append-only)

### Template (copy/paste)

1) Decide the output directory (must exist):

- `OUT_DIR=results/reports/<date>/<name>`  (or any run directory you use)
- `mkdir -p "$OUT_DIR"`

2) Run your command inside `tmux`, teeing logs:

- `COMMAND='pixi run <your-command-here>'`
- `bash -lc "$COMMAND" 2>&1 | tee -a \"$OUT_DIR/run.log\"`

### Monitor without tmux

- `tail -n 200 "$OUT_DIR/run.log"`
- `tail -f "$OUT_DIR/run.log"`

### If you restart a run

Keep `tee -a` so logs append; optionally add a timestamp separator:

- `printf \"\\n===== %s =====\\n\" \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" | tee -a \"$OUT_DIR/run.log\"`

