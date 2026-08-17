import os
DATA_DIR = os.environ.get("DTM_DATA_DIR", "F:/DTMModelData")
import pandas as pd
from pathlib import Path

for split in ["val", "train"]:
    f = Path(f"" + DATA_DIR + "/{split}/{split}_dataset.csv")
    if f.exists():
        df = pd.read_csv(f)
        print(f"{split}: {len(df):,} scenes  ({f.stat().st_size/1024/1024:.1f} MB)")
        print(f"  Titles: {dict(df['title'].value_counts())}")
    else:
        print(f"{split}: not started yet")
