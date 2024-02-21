import json
#import vessl

#vessl.init()

print('working')
# code snippet to transfer codes 
# Open both files
with open('data/sst5/train_sst5.jsonl', 'r') as f2, open('data/sst5/train_spaced_sst5_og.jsonl', 'r') as f1:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

# Ensure both files have the same number of lines
assert len(lines1) == len(lines2), "Files have different number of lines"

# Column to append from file2 to file1
column_to_append = ['label', 'label_text'] 

# Create a new list to hold the updated rows
new_rows = []

for line1, line2 in zip(lines1, lines2):
    # Load the JSON objects from the lines
    json1 = json.loads(line1)
    json2 = json.loads(line2)

    # Append the column from file2 to the JSON object from file1
    for col in column_to_append:
        json1[col] = json2[col]

    # Add the updated JSON object to the new rows
    new_rows.append(json1)

# Write the new rows to a new file
with open('data/sst5/train_spaced_sst5_og.jsonl', 'w') as f:
    for row in new_rows:
        f.write(json.dumps(row) + '\n')


'''

import json
import pandas as pd
import numpy as np

columns = ['0', '1', '2', '3', '4']

def rec_softmax(x):
    print(x)
    e_x = np.exp(x)
    e_x = 1/e_x
    return e_x / e_x.sum(axis=0)

def process_jsonl(file_path, output_file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    df[columns] = df[columns].apply(rec_softmax, axis = 1)

    df.to_json(output_file_path, orient='records', lines=True)
    return df

df = process_jsonl('data/sst5/train_spaced_sst5.jsonl', 'data/sst5/train_spaced_sst5.jsonl')
'''