import pandas as pd
from tqdm import tqdm

professions = ['Student', 'Technology', 'Arts', 'Education',
               'Communications-Media', 'Non-Profit', 'Engineering']

df = pd.read_csv('data/training_blogs_data.csv', sep='|')
new_df = None
max_samples = 15000
for class_ in tqdm(professions):
    temp_df = df.loc[df['Profession'] == class_]
    temp_df = temp_df.sample(frac=1).reset_index(drop=True)
    temp_df = temp_df.head(max_samples)
    try:
        new_df = new_df.append(temp_df, ignore_index=True)
    except Exception:
        new_df = temp_df

new_df = new_df.sample(frac=1).reset_index(drop=True)
new_df.to_csv('data/profession_dataset.csv', sep='|')
