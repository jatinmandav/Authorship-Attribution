import os
from tqdm import tqdm

from bs4 import BeautifulSoup

#text_file = open('raw_blog_text.txt', 'w')

for dir_ in ['data/training', 'data/test']:
    csv_file = open('{}_blogs_data.csv'.format(dir_), 'w')
    csv_file.write('Gender|Age_Group|Profession|Post\n')

    for xml_file in tqdm(os.listdir(dir_)):
        gender, age, profession = xml_file.split('.')[1:4]
        if int(age) < 20:
            age_group = '10'
        elif int(age) < 30:
            age_group = '20'
        else:
            age_group = '30'

        with open(os.path.join(dir_, xml_file),'rb') as fp:
            soup = BeautifulSoup(fp, 'xml')
            post = soup.find_all('post')
            for p in post:
                line = p.text
                line = line.replace('\t', '')
                line = line.replace('\n', '')
                line = line.replace('urlLink', '')
                line = line.replace('  ', '')
                line = line.replace('|', ' ')

                csv_file.write('{}|{}|{}|{}\n'.format(gender, age_group, profession, line))

#text_file.close()
csv_file.close()
