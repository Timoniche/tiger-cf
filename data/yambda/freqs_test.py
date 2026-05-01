
import json

import numpy as np
import torch


inter_json_path = '/Users/timoniche/IdeaProjects/tiger-cf/data/yambda/inter.json'
with open(inter_json_path, 'r') as f:
    user_interactions = json.load(f)

num_items = 0
freqs = {}
for items in user_interactions.values():
    for it in items:
        num_items = max(num_items, it + 1)
    for it in items[:-2]:
        freqs[it] = freqs.get(it, 0) + 1

freq_arr = np.zeros(num_items, dtype=np.int64)
for it, c in freqs.items():
    freq_arr[it] = c

item_freqs = torch.from_numpy(freq_arr)
cold_mask = ((item_freqs >= 5) & (item_freqs <= 10))
warm_mask = ((item_freqs >= 5) & (item_freqs <= 20))
hot_mask = ((item_freqs >= 5))

# num_items=33029, cold=17607, warm=25070, hot=32376
print(f'num_items={num_items}, cold={int(cold_mask.sum())}, warm={int(warm_mask.sum())}, hot={int(hot_mask.sum())}')


