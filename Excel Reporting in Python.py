#Import necessary Libraries
import pandas as pd
import numpy as np



#Import required datasets
df1 = pd.read_csv(r'path with file', encoding='latin-1', low_memory=False, error_bad_lines=False)
df2 = pd.read_excel(r'path with file', sheet_name = 'sheetname')
df3 = pd.read_csv(r'path with file', encoding='latin-1', low_memory=False, error_bad_lines=False)




#Get required columns from df1 master dataset and 
#Clean up suffixes to unify products of same family and remove duplicates
#Get Variable_1 and products from df1
df_filter = df1[['Variable_3', 'Variable_2', 'Variable_1','Variable_4']].copy()
df_filter['Variable_4'] = df_filter['Variable_4'].astype(str)
df_filter['Variable_4F'] = df_filter.Variable_4.str.replace(r'(-R|-U|-UU|-H|-ROHS|-S|U|UU)?$', '')
df_filter = df_filter.drop('Variable_4', axis = 1)
df_filter.columns = ['Variable_3', 'Variable_2', 'Variable_1', 'Variable_4']

df1_Variable_1 = df_filter[['Variable_1', 'Variable_4']].copy()
df1_Variable_1 = df1_Variable_1.drop_duplicates(how = 'any')






# Take online data for Region_Name and non-Condition_1 type only
# Take Item from online and find corresponding Sub type mapping from df1
df2 = df2[df2['type'] != 'Condition_1']
online = df2[df2['Condition_2'] == 'Region_Name'][['Region_Nametomer','Variable_2', 'Variable_3']].copy()
online = online.droVariable_4(how = 'any')
online.columns = ['Variable_3', 'Variable_1', 'Variable_4']

online['Variable_4'] = online['Variable_4'].astype(str)
online['Variable_4F'] = online.Variable_4.str.replace(r'(-R|-U|-UU|-H|-ROHS|-S|U|UU)?$', '')
online = online.drop('Variable_4', axis = 1)
online.columns = ['Variable_3', 'Variable_1', 'Variable_4']
online = online.drop_duplicates()
online = online.reset_index(drop = True)

df_act = pd.merge(online, df_filter[['Variable_2', 'Variable_4']], on = 'Variable_4', how = 'left')
df_act = df_act.droVariable_4(how = 'any')

dictfunc = {'Variable_3' : 'first',
            'Variable_1' : 'first',
            'Variable_2' : 'first'}

ds = df_act.groupby(['Variable_4']).agg(dictfunc)
ds['Variable_4'] = ds.index
ds = ds.reset_index(drop = True)

cols = ds.columns.tolist()
n = int(cols.index('Variable_1'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
ds = ds[cols]

cols = ds.columns.tolist()
n = int(cols.index('Variable_4'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
ds = ds[cols]

#Take list of products at Variable_1 in online data
online = online[['Variable_1', 'Variable_4']]
online.isnull().sum()
online = online.droVariable_4(how = 'any')
online.columns = ['Variable_1', 'Variable_4']

#Clean up suffixes to unify products of same family and remove duplicates
online['Variable_4'] = online['Variable_4'].astype(str)
online['Variable_4F'] = online.Variable_4.str.replace(r'(-R|-U|-UU|-H|-ROHS|-S|U|UU)?$', '')
online = online.drop('Variable_4', axis = 1)
online.columns = ['Variable_1', 'Variable_4']
online = online.drop_duplicates()
online = online.reset_index(drop = True)




#Clean df3 data
df3 = df3[['Variable_1', 'Part.Number']].copy()
df3.isnull().sum()
df3 = df3.droVariable_4(how = 'any')
df3.columns = ['Variable_1', 'Variable_4']

#Clean up df1 suffixes to unify products of same family and remove duplicates
df3['Variable_4'] = df3['Variable_4'].astype(str)
df3['Variable_4F'] = df3.Variable_4.str.replace(r'(-R|-U|-UU|-H|-ROHS|-S|U|UU)?$', '')
df3 = df3.drop('Variable_4', axis = 1)
df3.columns = ['Variable_1', 'Variable_4']
df3 = df3.drop_duplicates()
    




#Merge all df3, df1 and Online Variable_1_with_Item datasets
df_complete = df1_Variable_1.append(df3)
df_complete = df_complete.append(online)
df_complete.isnull().sum()
df_complete = df_complete.drop_duplicates()


df_Variable_1_products = df_complete.copy()


#Get the complete list of serviceable products at Variable_1 level
df_complete = df_complete.groupby(['Variable_1'])['Variable_4'].unique()
df_complete = pd.DataFrame(df_complete)
dfc = df_complete.copy()




#########################################




#Get the actual list of products bought by Region_Name at Variable_1 and Variable_4
df_filter = df_filter.drop_duplicates()
df_actual = df_filter.copy()
df_actual = df_actual.append(ds)
df_actual = df_actual.drop_duplicates()

df_actual.isnull().sum()
df_actual = df_actual[df_actual['Variable_1'].notnull()]
df_actual['Variable_3'] = df_actual['Variable_3'].astype(str) + '__' + df_actual['Variable_2'].astype(str)
dfa = df_actual.groupby(['Variable_3', 'Variable_1'])['Variable_4'].unique()
dfa = pd.DataFrame(dfa)
dfa.columns = ['Variable_4']
df_actual.columns = ['Variable_3', 'Variable_2', 'Variable_1', 'Variable_4', 'Variable_3']

#Get the unique combination of Variable_3, Variable_4 and Variable_1 and setting it as index
df_actual_bkp = df_actual.copy()
df_actual_bkp = df_actual_bkp.drop('Variable_4', axis=1)
df_temp = df_actual_bkp.drop_duplicates(subset= ['Variable_3', 'Variable_1'], keep='last')
df_temp = df_temp.set_index(['Variable_3', 'Variable_1'], drop = False)

dfact = pd.concat((df_temp, dfa), axis=1)



#Set Variable_1 as index in actual consumption dataframe so as to compare with whole 
#list at Variable_1 level
dfact = dfact.set_index(['Variable_1'], drop = False)



#Merge actual and complete dataframes on Variable_1 (stored as index)
ds = dfact.merge(dfc,how='left', right_index=True, left_index=True)



#Find products opportunity
ds['DIF'] = 'T'
for i in range(0, ds.shape[0]):
    ds['DIF'].iloc[i] = list(set(ds['Variable_4'].iloc[i]) - set(ds['Variable_4'].iloc[i]))
    


#Select the colums of interest for opportunity extract
extract = ds[['Variable_3', 'Variable_1', 'DIF']].copy()
extract = extract.reset_index(drop = True)



dn = ds[['Variable_3', 'Variable_1', 'DIF']].copy()

alist = []
for i in range(0, dn.shape[0]):
    alist = alist + dn.iloc[i, 2]
    alist = list(set(alist))




apd = pd.DataFrame({'products' : alist})
apd.to_csv(r'path to save file')




#Convert python list to comma seprated list of products
def listfunc(x):
    liststring = ''
    for i in range(0, len(x)):
        if i == 0:
            liststring = str(x[i])
        else:
            liststring = liststring + ', ' + str(x[i])
    return liststring

extract['Opportunity_Products'] = extract['DIF'].apply(lambda x: listfunc(x))

extract = extract.drop('DIF', axis = 1)




#Split the combined column "Variable_3" to separate the variables
extract[['Adf3ount', 'type']] = extract['Variable_3'].str.split('__', n=2, expand=True)



# Rearrange the columns
cols = extract.columns.tolist()
n = int(cols.index('Variable_1'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
extract = extract[cols]

cols = extract.columns.tolist()
n = int(cols.index('Opportunity_Products'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
extract = extract[cols]



extract['Count'] = extract['Opportunity_Products'].apply(lambda x: len(x))


#Putting opportunity products in separate columns
max_length = 0
for i in range(0, extract.shape[0]):
    x = len(list(extract['Opportunity_Products'][i].split(', ')))
    if x > max_length:
        max_length = x

item_cols = []
for i in range(1, max_length+1):
    item_cols.append('Item' + '_' + str(i))

extract[item_cols] = extract['Opportunity_Products'].str.split(', ', n=max_length, expand=True)


#Download the opprtunity products list - separated in columns
extract.to_csv(r'path to save the file')


