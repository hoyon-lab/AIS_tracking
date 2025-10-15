import pandas as pd
import numpy as np
df = pd.read_csv("./../dataset/AIS_2024_01_01/filtered_ais_nyc_classA_tracks.csv")
feature_cols = ["LAT", "LON", "SOG", "COG", "Heading"]
mmsi_groups = df.groupby("MMSI")
MIN_SEQ_LEN = 50
FIXED_SEQ_LEN = 50
all_sequences = []
for mmsi, group in mmsi_groups:
    group = group.sort_values(by="BaseDateTime")
    if len(group) < MIN_SEQ_LEN:
        continue  
    data = group[feature_cols].values
    for i in range(0, len(data) - FIXED_SEQ_LEN + 1):
        seq = data[i:i + FIXED_SEQ_LEN]
        all_sequences.append(seq)
print(f"총 시퀀스 수: {len(all_sequences)}")
