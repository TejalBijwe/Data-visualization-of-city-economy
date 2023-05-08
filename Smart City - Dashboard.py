import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random
import plotly.express as px
import random
from PIL import Image

import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------        
df = pd.read_csv(r"C:\Users\Owner\Desktop\Project\Dataset.csv")
df_d = df.copy()
df['date'] = df['date'].astype('str')

st.title('Smart City - Dashboard')

st.subheader('Dataset')
st.dataframe(df)
st.subheader('Data Numerical Statistic')
st.dataframe(df.describe())

st.subheader('Correlation heatmap of the data')
corr_mat = df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_mat, ax=ax)
st.write(fig,use_container_width=True)
st.write(' >>-- **Here**, we see that none of the features are highly correlated, which makes our data good to go for the Model Training.')

# st.subheader('Boxplots  - Looking for outliers')
# for clm in df.select_dtypes('number').columns:
#     if clm not in ['participantId','date']:
#         try:
#             fig = px.box(df, y=clm)
#             st.plotly_chart(fig, use_container_width=True)
#             pass
#         except:
#             print("plot can't be drawn for : ", clm)
# st.write("Looking at the outliers we find that these columns "rerecreation, rentadjustment, shelter, wage" have good amount of outliers present into them.
# Since all the 4 features are connected to the basic needs of a human, we are going to remove outliers but to an extent. Removing all the outliers would 
# make our data unrealistic. ")

st.subheader('Regression Model - Predicting Joviality')
le = LabelEncoder()
st.write('>>1. We are using Label Encoder for the categorical data to convert them into numerics. \n '
'>>2. The train test split is : Train size = 0.8 , Test size = 0.2  \n'
'>>3. To train our model, we are using RandomForestRegressor with 40 estimators.')
copydf = df
copydf.interestGroup = le.fit_transform(copydf.interestGroup)
copydf.haveKids = le.fit_transform(copydf.haveKids)

copydf1 = copydf.copy()
copydf1['date'] = copydf1['date'].astype('str')
copydf = copydf.drop(columns= ['participantId','date'])
st.write()
df_y = copydf['joviality']
X_train, X_test, y_train, y_test = train_test_split(copydf.drop('joviality',axis=1),df_y,test_size=0.2)

st.write()
model_reg = RandomForestRegressor(n_estimators=40)
model_reg.fit(X_train, y_train)

st.write('The score of our model is : ',model_reg.score(X_test,y_test))
y_predicted = model_reg.predict(X_test)
            

st.subheader('Classification model - Predicting Financial Status')

df_jov = copydf1.copy()
path = r"C:\Users\Owner\Desktop\Project\DataSet\Activity Logs\ParticipantStatusLogs"
all_als = {}
for i in [1,2,6,7,72]:
    df = pd.read_csv(path + str(i)+".csv")
    all_als["al_"+str(i)] = df
st.write("The activity logs of the people have the data of their activities within a fixed duration. It records their financial status after each activity to see if they are financially stable or not.")
cdf = pd.DataFrame()
st.write('>> Out of 72 datasets having records of their financial status, we will select only the datasets that contain stable and unstable both categories. \n Here are the Suffixes of each dataframe that satisfies the abovesaid condition.')
mls = []
for key in ['al_1','al_2','al_6','al_7','al_72']:
    if len(all_als[key]['financialStatus'].unique()) > 1:
        cdf = cdf.append(all_als[key])
        mls.append(key)
st.write("df suffixes are : ",str(mls))
        
cdf_2 = cdf.copy()

tomdf = cdf[['timestamp','participantId','financialStatus']]
tomdf = tomdf.rename(columns={'timestamp':'date'})
tomdf['date'] = tomdf['date'].astype('str')
tomdf['date'] = tomdf['date'].str[:10]
tomdf['date'] = tomdf['date'].str.replace("-","")

st.write('The length of activity logs data is : ',len(tomdf))
st.write('Where the unique counts of stable vs. unstable is : ', tomdf['financialStatus'].value_counts())
        
st.write(">> There is a third category in financial status , **Unknown**. So we are Putting this Unknown category of financial status into the 'Unstable' category")
tomdf['financialStatus'] = tomdf['financialStatus'].replace({'Unknown':'Unstable'}) 

st.write('Now unique counts of stable vs. unstable is : ',tomdf['financialStatus'].value_counts())

mgdf = copydf1.merge(tomdf, on=['participantId','date'], how='left').drop_duplicates()
df_fin = mgdf.copy()

#fig, ax = plt.subplots()
#ax.set_title('Heatmap to check null values present')
#sns.heatmap(mgdf.financialStatus.isnull(), ax=ax)
#st.write(fig,use_container_width=True)

na_cnt= mgdf.financialStatus.isnull().sum()

mgdf_nn = mgdf[mgdf.financialStatus.notnull()].reset_index().drop(columns=['index'])
mgdf_in = mgdf[mgdf.financialStatus.isnull()].reset_index().drop(columns=['index'])

mgdf_nn_1 = mgdf_nn.copy()
mgdf_nn = mgdf_nn.drop(columns=['participantId','date','joviality']).dropna()
mgdf_nn = mgdf_nn.dropna()

y_dat = mgdf_nn['financialStatus']
X_train, X_test, y_train, y_test = train_test_split(mgdf_nn.drop(columns=['financialStatus']),y_dat,test_size=0.2)
st.write('>>2. The train test split is : Train size = 0.8 , Test size = 0.2  \n'
'>>3. To train our model, we are using RandomForestClassifier with 2 estimators. The reason to use less estimators is to prevent our model from overfitting.')
model_cla = RandomForestClassifier(n_estimators=2)
model_cla.fit(X_train, y_train)
st.write('The score of the model is : ',model_cla.score(X_test, y_test))


#predicting the financial status for the old data where fiancial status was null (This null-not null split was done before training the model)
old_topred_data = mgdf_in.drop(columns=['participantId','date','joviality','financialStatus']).dropna()
pred_old_topred_data = pd.DataFrame({"y_pred_old_topred_data":model_cla.predict(old_topred_data)})
mgdf_in['financialStatus'] = pred_old_topred_data['y_pred_old_topred_data']

final_old_data = pd.concat([mgdf_nn_1 , mgdf_in])
nfsv = final_old_data.financialStatus.isnull().sum()


efr = pd.DataFrame({'Original Data :': [na_cnt]})
efr2= efr .to_string(index=False)
st.write(efr2)


st.write('Unique counts of financial status are : ', final_old_data.financialStatus.value_counts())
fut_data = final_old_data.copy()

st.subheader("Curating Data - Generating Future Data")
gfd_df = fut_data.copy()
gfd_df.drop(columns=['joviality','financialStatus'],inplace=True)
cat_data = gfd_df[['participantId','householdSize','haveKids','age','educationLevel','interestGroup']]
cur_cat_data = cat_data.groupby(['participantId']).mean().reset_index()
#Categorical data has been generated for the dataframe.

#Generating Continuous data based on calculating the CAGR and then estimating the coming year data
ddfin = gfd_df[['participantId','date','Education','Food','Recreation','RentAdjustment','Shelter','Wage']]
ddfin['date'] = ddfin['date'].str[:6]
ddfin = ddfin.groupby(['participantId','date']).sum().reset_index()

st.write('>> To curate the future data. We are using CAGR (Cumulative Annual Growth Rate) as our basis to calculate the particulars for next year. Though this is not the best idea, this will give us good results as the CAGR is generally applicable to all participants.')
#Calculating CAGR
cur_df = pd.DataFrame()
pids = ddfin['participantId'].unique()
for ppt in pids:
    wdf = ddfin[ddfin['participantId'] == ppt]
    wdf =  wdf.reset_index()
    wdf = wdf.drop(columns = ['index'])
    for c in wdf.columns:
        if c not in ['participantId', 'date']:
            cagr = (wdf[c][len(wdf)-1] / wdf[c][0])**(1/len(wdf)) - 1
#             print('cagr is at ', cagr, ": ", c, ": id is", ppt)
            wdf[c+"_cagr"] = cagr
    cur_df = cur_df.append(wdf)

cur_df_1 = cur_df.copy()

#Calculating final particulars with help of calculated CAGR
dfcols = ['participantId', 'date', 'Education', 'Food', 'Recreation','RentAdjustment', 'Shelter', 'Wage']
for mcol in dfcols:
    if mcol not in ['participantId', 'date']:
        cur_df[mcol] = cur_df[mcol] + cur_df[mcol]*cur_df[mcol+"_cagr"]
        
cur_df.fillna(0, inplace=True)
cur_df22 = cur_df[['participantId','date','Education', 'Food', 'Recreation','RentAdjustment', 'Shelter', 'Wage']]

#Keeping the expenditures in absolute numbers so that better visualisations can be drawn.
for mcol in cur_df22.columns:
    if mcol not in ['participantId', 'date']:
        cur_df22[mcol] = cur_df22[mcol].abs()

cur_fin_Data= pd.merge(cur_df22, cur_cat_data , on=['participantId'])
cur_fin_Data.head()
asdfsafd = pd.merge(cur_cat_data, cur_df22 , on='participantId')
st.write('This below is our curated data along with the calculated **CAGR**')
st.dataframe(asdfsafd)

joviality_pred = model_reg.predict(asdfsafd.drop(columns=['participantId','date']))
finstatus_pred = model_cla.predict(asdfsafd.drop(columns=['participantId','date']))
asdfsafd['joviality_pred'] = joviality_pred
asdfsafd['finstatus_pred'] = finstatus_pred

#Merging spends that are counted in expenses
asdfsafd['Expenses']  = 0
for cc in ['Education','Food','Recreation','RentAdjustment','Shelter']:
    asdfsafd['Expenses'] = asdfsafd['Expenses'] + asdfsafd[cc]

#Changing the dates of curated data to the next year (THis is necessary because we are not curating the data based on time series)
asdfsafd['date'] = asdfsafd['date'].str.replace('2023','2024')
asdfsafd['date'] = asdfsafd['date'].str.replace('2022','2023')
predicted = asdfsafd.copy()
edu_levels = {1:'Low',2:'HighSchoolOrCollege', 3:'Bachelors', 4:'Graduate'}
have_kids = {1:'True',2:'False'}
predicted['educationLevel'].replace(edu_levels, inplace=True)
predicted['haveKids'].replace(have_kids, inplace=True)

#Wages vs. Expenses Plot
fig = px.scatter(predicted, x='Wage', y='Expenses',animation_frame='date', animation_group='participantId',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs. Expenses Plot - Combined')
st.plotly_chart(fig, use_container_width=True)
# fig = px.scatter(predicted, x='Wage', y=['Education','Food','Recreation','RentAdjustment','Shelter'],animation_frame='date', animation_group='participantId',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs Expenses Plot - Individual parrticulars')
# st.plotly_chart(fig, use_container_width=True)

categs = ["0-18 group","19-25 group", "26-35 group", "36-50 group", "50-70 group", "70+ group"]
predicted.loc[(predicted.age <= 18),  'age_group'] = categs[0]
predicted.loc[((predicted.age > 18) & (predicted.age <= 25)),  'age_group'] = categs[1]
predicted.loc[((predicted.age > 25) & (predicted.age <= 35)),  'age_group'] = categs[2]
predicted.loc[((predicted.age > 35) & (predicted.age <= 50)),  'age_group'] = categs[3]
predicted.loc[((predicted.age > 50) & (predicted.age <= 70)),  'age_group'] = categs[4]
predicted.loc[(predicted.age >70),  'age_group'] = categs[5]

# Drawing some plots based on the various categories available
fig = px.scatter(predicted, x='Wage', y='Expenses',animation_frame='date', animation_group='participantId',color='educationLevel',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs. Expenses scatter plot (Education Level)')
st.plotly_chart(fig, use_container_width=True)
fig = px.scatter(predicted, x='Wage', y='Expenses',animation_frame='date', animation_group='participantId',color='age_group',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs. Expenses scatter plot (Age)')
st.plotly_chart(fig, use_container_width=True)
fig = px.scatter(predicted, x='Wage', y='Expenses',animation_frame='date', animation_group='participantId',color='haveKids',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs. Expenses scatter plot (Have kids)')
st.plotly_chart(fig, use_container_width=True)
fig = px.scatter(predicted, x='Wage', y='Expenses',animation_frame='date', animation_group='participantId',color='householdSize',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs. Expenses scatter plot (House hold size)')
st.plotly_chart(fig, use_container_width=True)
fig = px.scatter(predicted, x='Wage', y='Expenses',animation_frame='date', animation_group='participantId',color='interestGroup',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs. Expenses scatter plot (interest group)')
st.plotly_chart(fig, use_container_width=True)

st.write(">> Here, we see that all the plots are scattered in a legitimate way of living. \n There are a few things to note from each plot, as below: \n >> 1. Education level - the lower the education level, the lower will be the wages, so as the expenses. However, we see that the expenses vary to the full range regardless of their low wages. \n >> 2. Age Group  - In this plot, wages and expenses are both scattered. We don't see any specific pattern. That also shows that in real life, all age people earn and spend money throughout the range. \n >> 3. Have Kids - Here, we see an interesting pattern which is the obvious thing. Those who don't have kids are shifted to lower wages and lower expenses and vice versa. \n >> 4. household size - Here as well, the lower the household size, the lower will be the expenses and the wages.")

#fig = px.scatter(predicted, x='Wage', y='Expenses',animation_frame='date', animation_group='participantId',color='age_group',range_x=[1000,16000], range_y=[-100,3000], title='Wage vs. Expenses scatter plot based on age groups')
#st.plotly_chart(fig, use_container_width=True)

pred_W_E = predicted[['date','Wage','Expenses']].groupby('date').mean()
copydf1['Expenses']  = 0
for cc in ['Education','Food','Recreation','RentAdjustment','Shelter']:
    copydf1[cc] = copydf1[cc].abs()
    copydf1['Expenses'] = copydf1['Expenses'] + copydf[cc] 

copydf1['date'] = copydf1['date'].str[:6]
ordf = copydf1[['participantId','date','Wage','Expenses']].groupby(['participantId','date']).sum().abs()
ordf = ordf.reset_index()
orig_W_E = ordf.groupby('date').mean()
orig_W_E =orig_W_E.reset_index()
pred_W_E = pred_W_E.reset_index()
orig_W_E_1 = orig_W_E[~orig_W_E['date'].isin(['202303','202304','202305'])]
W_E_1 = pd.concat([orig_W_E_1,pred_W_E])

colors=['Predicted' if val > '202302' else 'Original' for val in W_E_1['date']]
W_E_1['date'] = W_E_1['date'].astype('O')

###########################################Do not remove###################
#fig = px.line(W_E_1 , x='date', y=['Wage','Expenses'],markers=True, title='Plot showing the Wages vs Expenses of original and Curated Data')
#fig.update_layout(plot_bgcolor="white", yaxis_title="Amount")
#for i, d in enumerate(fig.data):
#    fig.add_scatter(x=[d.x[-1]], y = [d.y[-1]],
#                    mode = 'markers+text',
#                   text = np.round(d.y[-1], 2),
#                    textfont = dict(color=d.line.color),
#                    textposition='middle right',
#                    marker = dict(color = d.line.color, size = 12),
#                    legendgroup = d.name,
#                    showlegend=False)
#st.plotly_chart(fig, use_container_width=True)
###################

image = Image.open(r"C:\Users\Owner\Desktop\Project\Plot.jpeg")
st.image(image, caption='Plot showing the Wages vs. Expenses of original and Curated Data')

# fig = px.line(W_E_1 , x='date', y=['Wage','Education','Food','Recreation','RentAdjustment','Shelter'],markers=True, title='Plot showing the Wages vs Expenses of original and Curated Data')
# fig.update_layout(plot_bgcolor="white", yaxis_title="Amount")
# for i, d in enumerate(fig.data):
#     fig.add_scatter(x=[d.x[-1]], y = [d.y[-1]],
#                     mode = 'markers+text',
#                     text = np.round(d.y[-1], 2),
#                     textfont = dict(color=d.line.color),
#                     textposition='middle right',
#                     marker = dict(color = d.line.color, size = 12),
#                     legendgroup = d.name,
#                     showlegend=False)
# st.plotly_chart(fig, use_container_width=True)

# Average joviality of the participants by age group
avg_jov_agegroup = asdfsafd.copy()
jov_pr = avg_jov_agegroup[['date','joviality_pred','age']].groupby(['date','age']).mean().reset_index()
jov_org = df_jov.copy()
jov_org['date'] = df_jov['date'].str[:6]
jov_org = jov_org[['date','joviality','age']].groupby(['date','age']).mean().reset_index()
jov_org = jov_org.drop(columns=['date'])
jov_pr = jov_pr.drop(columns=['date'])
ee = pd.merge(jov_org,jov_pr,on=['age']).drop_duplicates()
categs = ["0-18 group","19-25 group", "26-35 group", "36-50 group", "50-70 group", "70+ group"]
ee.loc[(ee.age <= 18),  'age_group'] = categs[0]
ee.loc[((ee.age > 18) & (ee.age <= 25)),  'age_group'] = categs[1]
ee.loc[((ee.age > 25) & (ee.age <= 35)),  'age_group'] = categs[2]
ee.loc[((ee.age > 35) & (ee.age <= 50)),  'age_group'] = categs[3]
ee.loc[((ee.age > 50) & (ee.age <= 70)),  'age_group'] = categs[4]
ee.loc[(ee.age >70),  'age_group'] = categs[5]

fig = px.bar(ee, x='age_group', y=['joviality','joviality_pred'],title="Average joviality of the participants by age group",barmode='group')
fig.update_layout(plot_bgcolor="white", yaxis_title="joviality")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# Average joviality of the participants by education Level
avg_jov_edu = asdfsafd.copy()
jov_pr_edu = avg_jov_edu[['date','joviality_pred','educationLevel']].groupby(['date','educationLevel']).mean().reset_index()
jov_org_edu = df_jov.copy()
jov_org_edu['date'] = jov_org_edu['date'].str[:6]
jov_org_edu = jov_org_edu[['date','joviality','educationLevel']].groupby(['date','educationLevel']).mean().reset_index()
jov_org_edu = jov_org_edu.drop(columns=['date'])
jov_pr_edu = jov_pr_edu.drop(columns=['date'])
eec = pd.merge(jov_org_edu,jov_pr_edu,on=['educationLevel']).drop_duplicates()
edu_levels = {1:'Low',2:'HighSchoolOrCollege', 3:'Bachelors', 4:'Graduate'}
eec['educationLevel'].replace(edu_levels, inplace=True)

fig = px.bar(eec, x='educationLevel', y=['joviality','joviality_pred'], title="Average joviality of the participants by education Level",barmode='group', text_auto=False)
fig.update_layout(plot_bgcolor="white", yaxis_title="joviality")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# Averages of the joviality - Existing data vs. Predicted data
ee_ovr = ee[['joviality','joviality_pred']].reset_index()
plt.bar(['joviality','joviality_pred'],[np.mean(ee_ovr['joviality']),np.mean(ee_ovr['joviality_pred'])])
plt.title('Averages of the joviality - Existing data vs. Predicted data')
plt.show()

# Counts of Stable and Unstable people
fdm = tomdf[['participantId','financialStatus']].dropna()
fdm = fdm.drop_duplicates()
cnt_ly = fdm['financialStatus'].value_counts()


ddsf = final_old_data[['participantId','financialStatus']].drop_duplicates()
cnt_ty = ddsf['financialStatus'].value_counts()

vcfd = pd.DataFrame({'category': ['Stable','Unstable'] ,'Last_Year':cnt_ly.to_list(), 'This_Year':cnt_ty.to_list()})
fig = px.bar(vcfd, x='category',y=['Last_Year','This_Year'],barmode='group',text_auto=True, title='Stable vs. Unstable counts of participants')
st.plotly_chart(fig, use_container_width=True)

st.write("This gives an inside stable and unstable participant prediction the following year. The number of stable participants is the same, but the unstable number is increasing, which can predict that the city is not progressing in terms of Financial health.")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.subheader('Travel Journal visualizations')

Jrnl_folder = r"C:\Users\Owner\Desktop\Project\DataSet\Journals\\"

##### 1. Travel Journal Data
df_TravelJournal = pd.read_csv(Jrnl_folder+"TravelJournal.csv")
df_TravelJournal['travelStartTime'] = pd.to_datetime(df_TravelJournal['travelStartTime'])
df_TravelJournal['travelEndTime'] = pd.to_datetime(df_TravelJournal['travelEndTime'])
df_TravelJournal['Duration(mins)'] = (df_TravelJournal['travelEndTime'] - df_TravelJournal['travelStartTime']).astype('timedelta64[m]')

trv_pd = df_TravelJournal[['participantId','travelStartTime','purpose', 'Duration(mins)']]
trv_pd['travelStartTime'] = trv_pd['travelStartTime'].astype('str')
trv_pd['travelStartTime'] = trv_pd['travelStartTime'].str[:7]
trv_pd_1 = trv_pd.groupby(['participantId','travelStartTime','purpose']).sum().reset_index()
trv_pd_1 = trv_pd_1.drop(columns= ['participantId']).groupby(['travelStartTime','purpose']).mean().reset_index()

fig = px.line(trv_pd_1 , x= 'travelStartTime', y = 'Duration(mins)', color='purpose', title="Average Travel time by purpose of travel")
fig.update_layout(plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

df_TravelJournal['Spends'] = df_TravelJournal['startingBalance'] - df_TravelJournal['endingBalance']
trv_sd = df_TravelJournal[['participantId','travelStartTime','purpose', 'Spends']]
trv_sd['travelStartTime'] = trv_sd['travelStartTime'].astype('str')
trv_sd['travelStartTime'] = trv_sd['travelStartTime'].str[:7]
trv_sd_1 = trv_sd.groupby(['participantId','travelStartTime','purpose']).sum().reset_index()
trv_sd_1 = trv_sd_1.drop(columns= ['participantId']).groupby(['travelStartTime','purpose']).mean().reset_index()
trv_sd_1['Spends'] = trv_sd_1['Spends'].abs()

fig = px.line(trv_sd_1 , x= 'travelStartTime', y = 'Spends', color='purpose', title="Average Spends by purpose of travel")
fig.update_layout(plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

st.write("These findings collectively support the conclusion that the city is facing financial challenges, as evident from the decreasing travel time, reduced spending on social gatherings, and increased time at home. The data suggest that participants have curtailed their travel and social activities, likely due to financial concerns, and have been spending more time at home.")





















