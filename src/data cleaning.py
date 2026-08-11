import pandas as pd
import re
data=pd.read_csv(r'D:\NLP project\data\vulnerabilities.csv')

#missing values
drop_null=data.dropna()

#duplicates
drop_duplicates=drop_null.drop_duplicates()

#drop columns
drop_columns=drop_duplicates.drop(columns=['commit_id', 'hash', 'project', 'message'])

#reduce the sample to 50000 rows according to the size of the source code
#best_data=drop_columns.nlargest(50000, 'size')

import numpy as np
target_1 = drop_columns[drop_columns['target'] == 1]
target_0_pool = drop_columns[drop_columns['target'] == 0].sort_values(by='size').reset_index(drop=True)
pool_sizes = target_0_pool['size'].values
selected_indices = []
used_indices = set()

for size in target_1['size'].values:
    idx = np.searchsorted(pool_sizes, size)
    best_idx = None
    best_diff = float('inf')
    for i in range(max(0, idx - 100), min(len(pool_sizes), idx + 100)):
        if i not in used_indices:
            diff = abs(pool_sizes[i] - size)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
    if best_idx is not None:
        used_indices.add(best_idx)
        selected_indices.append(best_idx)
    else:
        for i in range(len(pool_sizes)):
            if i not in used_indices:
                used_indices.add(i)
                selected_indices.append(i)
                break

target_0 = target_0_pool.iloc[selected_indices]
combined_data = pd.concat([target_0, target_1])

#remove comments
def remove_comments(text):
    if not isinstance(text, str):
        return ""
    # Remove single-line comments
    code = re.sub(r'//.*', '', text)
    # Remove multi-line comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code)
    return code.strip()

combined_data['func'] = combined_data['func'].apply(remove_comments)

#check the no.of total rows, no.of vulnerable and non-vulnerable rows
b=len(combined_data['func'])
c=len(combined_data[combined_data['target']==0])
d=len(combined_data[combined_data['target']==1])
print(f"Total: {b}, Safe: {c}, Vulnerable: {d}")

combined_data.to_csv(r'D:\NLP project\data\cleaned_data_of_vulnerabilities.csv', index=False)