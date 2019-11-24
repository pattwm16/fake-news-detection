# preprocessing
# Will Patterson
import pandas as pd

# DATA LAYOUT ----
# Column 1: the ID of the statement ([ID].json).
# Column 2: the label.
# Column 3: the statement.
# Column 4: the subject(s).
# Column 5: the speaker.
# Column 6: the speaker's job title.
# Column 7: the state info.
# Column 8: the party affiliation.
# Column 9-13: the total credit history count, including the current statement.
# 9: barely true counts.
# 10: false counts.
# 11: half true counts.
# 12: mostly true counts.
# 13: pants on fire counts.
# Column 14: the context (venue / location of the speech or statement).
# -----

# path to training dataset
train_path = "liar_dataset/train.tsv"
test_path = "liar_dataset/test.tsv"
valid_path = "liar_dataset/valid.tsv"

header_names = ['ID', 'Label', 'Statement', 'Subjects', 'Speaker',
                'Speaker Job', 'State info', 'Party Affiliation',
                'Barely true count', 'False count', 'Half true count',
                'Mostly true count', 'Pants on fire count', 'Context']

train_data = pd.read_csv(train_path, sep='\t', header=None, names=header_names)
test_data = pd.read_csv(test_path, sep='\t', header=None, names=header_names)
valid_data = pd.read_csv(valid_path, sep='\t', header=None, names=header_names)
