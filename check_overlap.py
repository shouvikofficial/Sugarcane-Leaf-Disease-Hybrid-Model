import os, hashlib
import pandas as pd
from pathlib import Path
import kagglehub
from sklearn.model_selection import train_test_split

SEED = 42
path = kagglehub.dataset_download("ingjuanquevedo/sugarcane-smut-dataset")
rows = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            rows.append((os.path.join(root, f), os.path.basename(root)))
df_all = pd.DataFrame(rows, columns=['filepath', 'label'])

# OLD split (no dedup) - the one Optuna used
_, tmp = train_test_split(df_all, test_size=0.30, stratify=df_all['label'], random_state=SEED)
old_val, old_test = train_test_split(tmp, test_size=0.50, stratify=tmp['label'], random_state=SEED)
print("Old split val/test:", len(old_val), len(old_test))

# NEW split (dedup) - the one the final model used
df_all['file_hash'] = df_all['filepath'].apply(lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest())
df_new = df_all.loc[~df_all.duplicated(subset='file_hash', keep='first')].copy()
tr, tmp2 = train_test_split(df_new, test_size=0.30, stratify=df_new['label'], random_state=SEED)
new_val, new_test = train_test_split(tmp2, test_size=0.50, stratify=tmp2['label'], random_state=SEED)
print("New split train/val/test:", len(tr), len(new_val), len(new_test))

old_val_hashes = set(df_all.loc[df_all['filepath'].isin(old_val['filepath']), 'file_hash'])
ov_test = set(new_test['file_hash']) & old_val_hashes
ov_val = set(new_val['file_hash']) & old_val_hashes
print("New TEST images that were in Optuna's old validation set:", len(ov_test), "of", len(new_test))
print("New VAL images that were in old validation set:", len(ov_val), "of", len(new_val))