```
uv venv --python 3.12
```
```
uv sync
```
```
uv pip install -e .
```
```
uv run storb-eval eval --submission smolvla_submission --task push-v3 --episodes 1 --workers 1 --max-steps 10 --render --fps 120 --render-mode human --json | cat
```
check debug_images for pics of what smolval is seeing
if you update deps of the submissions then delete .requirements_installed file