# CDR+-aligned

This is a runnable CDR+ project rebuilt from the complete original CDR codebase
and the CDR+ implementation fragment.

The model uses the projected type-specific embeddings for contrastive CD loss
and inference, adds centroid orthogonality and reconstruction losses, and keeps
the pre-trained weights when starting fine-tuning.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py --dataname mafengwo --lambda-ortho 0.5 --lambda-recon 0.5
```

Use `--dataname youshu` for the bundle-recommendation dataset. Results and
checkpoints are written below `save/<dataset>/<timestamp>/`.
