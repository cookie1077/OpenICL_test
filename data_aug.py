import json
import vessl

vessl.init()

# Open both files
with open('data/train.jsonl', 'r') as f1, open('data/train_newtest.jsonl', 'r') as f2:
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
with open('/output/train_newtest2.jsonl', 'w') as f:
    for row in new_rows:
        f.write(json.dumps(row) + '\n')