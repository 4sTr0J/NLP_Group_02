import pandas as pd
import re
data=pd.read_csv(r'G:\My Drive\NLP project\data\vulnerabilities.csv')

#missing values
drop_null=data.dropna()

#duplicates
drop_duplicates=drop_null.drop_duplicates()

#drop columns
drop_columns=drop_duplicates.drop(columns=['commit_id', 'hash', 'project', 'message'])

#reduce the sample to 50000 rows according to the size of the source code
#best_data=drop_columns.nlargest(50000, 'size')

target_0 = drop_columns[drop_columns['target'] == 0].sort_values(by='size', ascending=False).head(25000)
target_1 = drop_columns[drop_columns['target'] == 1].sort_values(by='size', ascending=False).head(25000)
combined_data = pd.concat([target_0, target_1])

#remove comments
def remove_comments(text):
    # Remove single-line comments
    code = re.sub(r'//.*', '', text)
    # Remove multi-line comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code)
    return code.strip()

combined_data['func']=drop_columns['func'].apply(remove_comments)

#check the no.of total rows, no.of vulnerable and non-vulnerable rows
# b=len(combined_data['func'])
# c=len(combined_data[combined_data['target']==0])
# d=len(combined_data[combined_data['target']==1])
# print(b,c,d)

#combined_data.to_csv(r'G:\My Drive\NLP project\data\cleaned_data_of_vulnerabilities.csv', index=False)