# Reconstructing repeated strict-split training CSVs

Large training matrices are stored as ordered Git LFS chunks.

For any condition folder:

```bash
cd data/processed_splits/repeated_component_resampling/<condition>

NAME="train_strict_unseen_solute_maccs_map4_padel_sen_features.csv"
cat "$NAME.chunks/$NAME".part-* > "$NAME"

sha256sum -c "$NAME.sha256"
```

For unseen-solvent folders, replace `solute` with `solvent` in `NAME`.

The corresponding test CSVs are stored directly through Git LFS.
