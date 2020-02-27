#!/usr/bin/env python
# coding: utf-8

# ## AcF 351: Textual analysis on 10-K filings from 2000-2018
# #### Does the sentiment of financial statement give an indication on subsequent short/medium/long-term share performance?
# #### Do firms that include more ESG topics in their financial statements outperform those that include less ESG topics?

# In[1085]:


# The following file will extract the SEC fillings from the EDGAR webpage for all the companies that have submitted a file
# to the SEC during a certain timeframe, 2000 to 2018 in this case. First, the full index links will be dowloaded, from recurring 
# 10-K files. Thereafter, only the direct link to 10-K files will be dowloaded as these are the ones of interest. 
# Based on the 10-K links, I will extract the full html files by withdrawing the tables and keeping only the full texts. 
# I clean the dataset by withdrawing stop words, certain key words as "company" or the years of interests or by withdrawing
# certain html tags. Finally, I count the numbers of positive and negative words in each 10-K file based on two different libraries: 
# the one from Loughran and McDonald (2016) and the one from Bing Liu. Finally, a polarity score is extracted based on the 
# number of positive and negative words from each 10-K file. 

# ATTENTION: The given code below is extremely computational intensive and therefore should not be exectued diretly without
# using an appropriate machine or slicing the workload in different steps.It is highly discouraged to exectue the code below for an entire
# timeframe on only one machine and it is adviced to cut the process by year to be able to run it on several machines 
# to be efficient. If the need arises that the code has to be run in one time, it is recommended to use the service of
# Google Cloud or Amazon Web Services (AWS)


# In[1086]:


# Import libraries needed

import os
import csv
import time
import random
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import parser
import timeit
import pickle
import datetime as dt
import wrds
import nltk
from nltk.tokenize import RegexpTokenizer, sent_tokenize
nltk.download('punkt')
import matplotlib.pyplot as plt
import pandas_profiling
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats



# In[1099]:


# set working directory
PATH = r'YOUR PATH HERE'
os.chdir(PATH)

# In my workings: PATH = '/Users/user/Documents/Uni 19:20/AcF351/Coursework'


# In[1091]:


# create new folders for this project
for dir in ['CSV files', 'Linking table', 'Output pickles', 'Dictionaries', 'Cleaned text', 'Textual Analysis scores','Analysis']:
    os.mkdir(dir)


# ## Download datasets for later use so that the program can be run in one go hereafter

# In[1155]:


os.chdir(PATH + r'/Dictionaries')


# #### Download sentiment words (LM + BL)  & stop words 

# In[1157]:


# Loughran & McDonald

# Stop words - https://sraf.nd.edu/textual-analysis/resources/#StopWords
# NOTE: take the generics file and save as a new file as follows: StopWords_Generic.txt

# Master dictionary - https://sraf.nd.edu/textual-analysis/resources/#LM%20Sentiment%20Word%20Lists
# NOTE: take the positive words column and save as a new file as follows: bl_positive.csv
# NOTE: take the negative words column and save as a new file as follows: bl_negative.csv


# Bing Liu provides an alternative to the LM dictionary - whose words have been more or less used (depending on meaning) 
# in financial statements by companies after their first papers appeared on textual analysis 
# (thus another dictionary may provide useful)

# Bing Liu datasets overview https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets

# Negative Bing Liu dictionary words - https://gist.github.com/mkulakowski2/4289441
# NOTE: take the negative words column and save as a new file as follows: lm_negative.csv
# Positive Bing Liu dictionary words -  https://gist.github.com/mkulakowski2/4289437
# NOTE: take the positive words column and save as a new file as follows: lm_positive.csv


# In[1158]:


os.chdir(PATH + r'/Linking table')


# #### Download industry classifications linking table  (has to be done manually)

# In[ ]:


# Dataset needed for industry classification - https://www.census.gov/eos/www/naics/downloadables/downloadables.html
# NOTE: Download the '6-digit 2012 Code File' and save as follows: 2017_NAICS_Structure.xlsx


# In[73]:


# automate url creation of SEC Index page

def getIndexURLs(start=2000, end=2018):
    urls = []
    for year in range(start, end+1):
        for qtr in ['QTR1', 'QTR2', 'QTR3', 'QTR4']:
            url = 'https://www.sec.gov/Archives/edgar/full-index/{}/{}/crawler.idx'.format(year, qtr)
            urls.append(url)
    return urls


# In[74]:


# check whether the above line of code is working

getIndexURLs(2000,2018)


# In[75]:


# Now we have a list of address, let's retrieve information from crawler.idx files
# and rewrite the data to csv locally
def writeIndexCSV(url, header_loc=7, firstrow_loc=9):# because the hyphons are separarting them
    r = requests.get(url)
    lines = r.text.splitlines()
    
    # retrieve the location of columns
    name_loc = lines[header_loc].find('Company Name')
    type_loc = lines[header_loc].find('Form Type')
    cik_loc = lines[header_loc].find('CIK')
    date_loc = lines[header_loc].find('Date Filed')
    url_loc = lines[header_loc].find('URL')
     
    # create file name based on the original idx file
    file_yr = url.split('/')[-3]
    file_qtr = url.split('/')[-2][-1]
    file_name = file_yr + "Q" + file_qtr + ".csv"
    
    # create and write to csv file
    with open(file_name, 'w') as wf:
        writer = csv.writer(wf, delimiter = ',')
        
        # go through lines
        for line in lines[firstrow_loc:]:
            company_name = line[:type_loc].strip()
            form_type = line[type_loc:cik_loc].strip()
            cik = line[cik_loc:date_loc].strip()
            date_filed = line[date_loc:url_loc].strip()
            page_url = line[url_loc:].strip()
            
            # let's foucs on 10-K files only
            if form_type == '10-K':
            
                # create a new row of data using tuple which is ordered and unchanged
                row = [company_name, form_type, cik, date_filed, page_url]
                writer.writerow(row)
                
        print("{} saved".format(file_name))


# In[1163]:


# save in the right folder
os.chdir(PATH + r'/CSV files')


# In[12]:


# transform idx crawler files into csv files

for url in getIndexURLs(2000,2018): # YEAR 2011 QUARTER 4 and 2017 Q2 are way slower than the others so just be patient when running the code
    writeIndexCSV(url)
    time.sleep(3 + random.random() * 3)


# In[90]:


# this function will put our already downloaded individual CSV files into one big pandas dataframe

# URL is link to the filing page IS NOT EQUAL to the financial statement

sub_df = []
columns = ["Company", "FS", "CIK", "Date", "URL"]

for year in range(2000, 2018):
    for qtr in ['Q1', 'Q2', 'Q3', 'Q4']:
        path = '/Users/User/CourseworkCSV/{}{}.csv'.format(year,qtr)                        
        csv  = pd.read_csv(path, names=columns)
        sub_df.append(csv)
        df = pd.concat(sub_df, ignore_index=True)
        


# In[282]:


# to see the whole line of text such that we can copy and paste into our browser to check whether urls are working
# and generally for larger dataframes
pd.set_option('display.max_colwidth', -1) 


# In[92]:


# representation of the above lines of code
# every 10-K overview filing url from the last 18 years in the US is in this table

df.head()


# In[93]:


# this function takes the whole pandas datframe, takes one column and converts it into a numpy array

# I am doing this because previously I was running a for loop through the dataframe which would have taken some days 
# to complete, after some research I found out that numpy speeds up the iteration process significantly 
# hence why I am doing these intermediate steps

# for reference: #  https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6

Intermediate_URLs = df['URL'].values


# In[94]:


# Parse 10K Form page, including 10-K form URL and other meta data

def parseFormPageTXTincl(url):
    '''
    Input: URL
    
    Output:
        filer_cik:
        filing_date:
        report_date:
        form_url
    ''' 
    # get page and create soup
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    
    # parse filer Info on 10K page
    filer_div = soup.find('div', {'id': 'filerDiv'})
    filer_text = filer_div.find('span', {'class': 'companyName'}).find('a').get_text()
    filer_cik = re.search(r"(\d{10})\s(\(.+\))$" ,filer_text)[1] # to get CIK number via search function
    
    # parse 10K Page Meta data
    form_content = soup.find('div', {'class': 'formContent'})
    
    filing_date = form_content.find('div', text='Filing Date').findNext('div').get_text()
    report_date = form_content.find('div', text='Period of Report').findNext('div').get_text()
    
    # check whether we have 10-K files (there are exceptions somehow)
    
    table = soup.find('table', {'class': 'tableFile', 'summary': 'Document Format Files'})
    IOK_check = str(table.find('td', text='10-K')) # returns -1 if not 10-K
    if IOK_check.find('10-K') == -1:
         print('non 10-K file') # because it is true it is not a 10-K filing and should net be included in our analysis
            
    else:
        table = soup.find('table', {'class': 'tableFile', 'summary': 'Document Format Files'})
        href = table.find('td', text='10-K').find_parent('tr').find('a')['href'] # find parent looks for the outer tr and then looks within the tr for a link
        form_url = "https://www.sec.gov" + href
    
    # now include IF statement to accomodate make a difference between the txt location and htm location
        
        if form_url.endswith('htm'):  # for 10-K htm documents
            return filer_cik, filing_date, report_date, form_url  
        
        
        else:  # to account for non htm 10-K files i.e. txt files 
            table = soup.find('table', {'class': 'tableFile', 'summary': 'Document Format Files'})
            href = table.find('td', text='Complete submission text file').find_parent('tr').find('a')['href']
            form_url = "https://www.sec.gov" + href
        
       #  if form_url.endswith('txt'):  # for 10-K txt documents
            return filer_cik, filing_date, report_date, form_url


# In[95]:


Intermediate_URLs


# In[96]:


# check to see whether we still have the same amount of datapoints as in the pandas dataframe

# problem in row 4252 because it is not only 10-K but 10-K405: https://www.sec.gov/Archives/edgar/data/701345/0000701345-00-000005-index.htm
# problem in row 727 because it is not only 10-K but 10Q: https://www.sec.gov/Archives/edgar/data/813920/0000813920-00-000001-index.htm

np.size(Intermediate_URLs) 


# In[97]:


FSThtml_Links = []

for row in Intermediate_URLs:
    FSThtml_Links.append(parseFormPageTXTincl(row))
    
# all in all we get 43 mistakes i.e. documents whose names have been changed from 10-K to 10-K405 for instance
# see above line for two examples why mistakes here occur
# given the small size of mistakes relative to all the observations these are neglectable
    


# In[ ]:


# NOTE: what follows is NOT the most efficient way: if I had to re-do I would use the pickle module


# In[ ]:


FSThtml_Links.to_csv('FSThtmlLinks.csv', encoding='utf-8', index=False)


# In[ ]:


# we are downloading the ouput so that we do not have to re-run the whole notebook/code when restarting the kernel 
# efficient when starting to work on data but if this code is run straight then saving Python output to csv/pickle is 
# not required necessarily (but recommended)


# In[1164]:


# read into Jupyter again
data2000_2018 = pd.read_csv('FSThtmlLinks.csv',header=None)


# In[1165]:


# have a look at the file
data2000_2018
# as we see the data is pasted horizontally which makes working with it uncomfortable


# In[1166]:


# doing this because FSThtml_Links is a numpy list (since it was the quickest way when doing many iterations)
# when you export a list into csv/excel it will be only 1 row of data which is difficult to handle
# so the solution is to take the transpose to get it back into the classical row shape
# now the data for once company is in 1 single column so we need to get it into 4 different colums

data2000_2018i = data2000_2018.transpose() # where i stands for intermediate format


# In[1167]:


# prepare the dataframe for subsequent steps
# need to split the initial_output into the respective columns

data2000_2018i.rename(columns={0: 'Initial_output'}, inplace=True)


# In[1168]:


data2000_2018i.head()

# now we have 4 colums which we fill with the right data contained in the most left column


# In[1169]:


data2000_2018ii = data2000_2018i['Initial_output'].str.split(',', n = 4, expand = True)


# In[1170]:


data2000_2018ii.head()


# In[1171]:


# creating a new dataset where we get the data in shape and the right format so that we can start working on it
# for example we could not run the url through our function as it was accompanies by ') which makes the url not work
# another handy feature of this is that we can for example click on the link in this workbook when we express 
# the Form_url column as a series
data2000_2018c = pd.DataFrame(columns = ['CIK','Filing_date', 'Report_date', 'Form_url'])

data2000_2018c['CIK'] = data2000_2018ii[0].str.slice(start=2,stop=-1)
data2000_2018c['Filing_date'] = data2000_2018ii[1].str.slice(start=2,stop=-1)
data2000_2018c['Report_date'] = data2000_2018ii[2].str.slice(start=2,stop=-1)
data2000_2018c['Form_url'] = data2000_2018ii[3].str.slice(start=2,stop=-2)

data2000_2018c.head()


# In[61]:


data2000_2018c.to_csv('Cleaned_Data2000_2018.csv', encoding='utf-8', index=False)

# now we have saved it with an appropriate name and appropriate format unlike the first saving we did
# could have modified that data first and then saved it only once but given that I changed computers it was the only way


# In[3]:


# doing this so that when we re-open the workbook and the kernel has to reload we do not have to run the above steps
# (could also be done with pickle but only learned about pickle at a later stage)


# In[1064]:


# load as text in order to keep the 0s in front of the first non-zero number
data2000_2018C = pd.read_csv('Cleaned_Data2000_2018.csv',dtype = {"CIK": object})


# In[1065]:


# need to change Filing date from 'object' type to 'datetime' object for merging later on
data2000_2018C['Filing_date']=pd.to_datetime(data2000_2018C['Filing_date'])

data2000_2018C.info()
data2000_2018C.head()


# ## Linking Textual analysis scorces (CIK) with Compustat (Govkey)

# In[1066]:


# TO MATCH COMPANIES ACROSS THE 2 DATASETS
# govkey in COMPUSTAT is equal to CIK in SEC

# TO MATCH ACROSS TIME PERIODS
# datadate in COMPUSTAT is equal to reporting date from SEC


# #### Useful to get particular observations given a certain CIK

# In[1177]:


data2000_2018C.loc[data2000_2018C['CIK'] == '0001318605']


# ### Connect to WRDS

# In[5]:


# WRDS Connection
conn = wrds.Connection()


# ### Define sample period

# In[6]:


# pre-define the sample periods
COMPUSTAT_BEG_DATE = '01/01/2000'
CRSP_BEG_DATE = '01/01/2000'
CRSP_END_DATE = '12/31/2017'


# ### Retrieve data (http://www.crsp.com/files/ccm_data_guide_0.pdf)

# In[7]:


# NEED TO GET THE INDUSTRY CLASSIFICATION INTO THE SQL SEARCH QUERY


# Postgres Query
stmt = """
          SELECT gvkey, datadate, at, pstkl, txditc, pstkrv, seq, pstk, cik, cusip
          FROM comp.funda
          WHERE indfmt='INDL' 
          AND datafmt='STD' 
          AND popsrc='D'
          AND consol='C'
          AND datadate >= '{}'
       """.format(COMPUSTAT_BEG_DATE)
comp = conn.raw_sql(stmt)

# INDFMT is the Industry Format code that indicates whether a company reports in a Industrial format (INDL)  
# DATAFMT is the Data Format code that indicates how the data is collected and presented, with Standardized (STD) being the common and useful type
# POPSRC indicates the Population Source of the data as Domestic (D) or International (I)
# CONSOL shows the Level of Consolidation that identifies whether a company's financial statements represent consolidated (C) or nonconsolidated information (N), or another type


# ### Work on Compustat data (https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-compustat-north-america-global-and-bank/)
# 

# In[8]:


# set date and time to the standard format recognised by Pandas and other packages
comp['datadate']=pd.to_datetime(comp['datadate'])

# create a new variable for year
comp['year']=comp['datadate'].dt.year


# In[9]:


# check data
comp.info()
comp.head()


# In[10]:


comp.loc[comp['gvkey'] == '012994']


# ### Data cleaning

# In[11]:


# keep only rows with shareprice and cik number
comp.dropna(subset=['gvkey','datadate','cik','cusip'])

comp.head()


# In[12]:


# sort values so that the dataframe is constructed by id and time
comp = comp.sort_values(by = ['gvkey','datadate']) # gvkey is the primary company id

# count obs? starting from 0 to length of the group -1
comp['count'] = comp.groupby(['gvkey']).cumcount()

# house cleanning
comp = comp[['gvkey','datadate','year','cik','cusip','count']]

comp[:10]


# ## Get financial data from CUSIP (majoritarily from Fama French codes)

# In[13]:


# Link for explanation of abbreviations and CRSP data structure: 
# http://www.crsp.com/products/documentation/stock-data-structure'


# #### Important definitions

# In[14]:


# ret = Month-end to month-end change in total investment of a security, with ordinary dividends reinvested at the month-end
# retx = Month-end to month-end capital appreciation of a security, calculate as a change in price only


# ## Getting daily share price data from CRSP

# In[15]:


# sql for returning a merged crsp price dataset
# its a large dataset, takes time to run (approx 15min since it is 2GB of data)
stmt = """
          SELECT a.permno, a.permco, a.date, a.ret, a.retx, a.shrout, a.prc,
                 b.comnam, b.naics
          FROM crsp.dsf AS a
          LEFT JOIN crsp.msenames AS b
          ON a.permno=b.permno
          AND b.namedt<=a.date
          AND a.date<=b.nameendt
          WHERE a.date BETWEEN '{}' AND '{}'
          AND b.exchcd BETWEEN 1 AND 3
       """.format(CRSP_BEG_DATE, CRSP_END_DATE)
crsp_d = conn.raw_sql(stmt)


# In[16]:


crsp_d.info()
crsp_d.head()


# In[17]:


# replace NA or INF with -1 to make the change from float64 to int64
crsp_d = crsp_d.fillna(-1)
# change variable format to int
crsp_d[['permco','permno','shrout']] = crsp_d[['permco','permno','shrout']].astype(int)


# In[18]:


# because on some occasions daily return given by CRSP differs from real daily return 
# also compute previous day,1 week, 1 month, 1 year return

# previous day return
crsp_d['n-1_to_n_ret'] = crsp_d.groupby(['permno'])['prc'].pct_change()
# next day return
crsp_d['n_to_n+1_ret'] = crsp_d.groupby(['permno'])['prc'].shift(-1).pct_change()
# 5 days return if published on a monday otherwise 7 days return
crsp_d['n_to_n+5_ret'] = crsp_d.groupby(['permno'])['prc'].pct_change(periods = 5).shift(-5)
# 1 month return (on average a month has 22 business days i.e. 22 days of stock prices)
crsp_d['n_to_1month_ret'] = crsp_d.groupby(['permno'])['prc'].pct_change(periods = 22).shift(-22)
# 1 year return (on average a year has 261 business days i.e. 261 days of stock prices)
crsp_d['n_to_1year_ret'] = crsp_d.groupby(['permno'])['prc'].pct_change(periods = 261).shift(-261)


# In[19]:


crsp_d['date'] = pd.to_datetime(crsp_d['date'])
crsp_d['jdate'] = crsp_d['date'].dt.year


# In[20]:


crsp_d.info()
crsp_d.head()


# ### Retrieving CCM data

# In[21]:


ccm=conn.raw_sql("""
                    SELECT gvkey, lpermno AS permno, linktype, linkprim, 
                    linkdt, linkenddt
                    FROM crsp.ccmxpf_linktable
                    WHERE SUBSTR(linktype,1,1)='L'
                    AND (linkprim ='C' OR linkprim='P')
                 """)


# In[22]:


# convert datetime
ccm['linkdt']=pd.to_datetime(ccm['linkdt'])
ccm['linkenddt']=pd.to_datetime(ccm['linkenddt'])
# if linkenddt is missing then set to today date
ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))


# In[23]:


ccm.head()


# In[24]:


ccm.loc[ccm['permno'] == 10001]


# ### Merge Compustat and CRSP

# #### Merge Compustat and linking table

# In[25]:


# left merge on gvkey
ccm1 = pd.merge(comp[['gvkey', 'datadate','year','cik','cusip','count']], ccm, how = 'left', on = ['gvkey'])
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0) # WHY IS THIS USEFUL?????
# create 'jdate' for further merge with crsp dataset
ccm1['jdate'] = ccm1['yearend'] 


# In[26]:


ccm1.info()
ccm1.head()


# #### Merge CRSP into the above created table

# In[27]:


# set link date bounds to delete entries which are not needed respectively do not respect below criteria
ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]
ccm2 = ccm2[['gvkey', 'permno', 'datadate', 'yearend', 'jdate','cik','cusip','year']]


# In[28]:


#ccm2.drop(ccm2.columns[11] ,axis = 1)
ccm2.info()
ccm2.head()


# In[29]:


# link comp and CRSP
ccm_complete = pd.merge(crsp_d, ccm2, how= 'inner', left_on= ['permno','jdate'], right_on= ['permno','year'])
# Changing cusip to CUSIP 9 (since compustat gives the 9 character cusip code)


ccm_complete['cusip 9'] = ccm_complete['cusip']
ccm_complete = ccm_complete.drop(columns="cusip")
# change to string for slicing in the next step
ccm_complete['cusip 9'] = ccm_complete['cusip 9'].astype(str)
# adding year so that we can merge the two dataframes on CUSIP 6 and year
ccm_complete['year'] = ccm_complete['date'].dt.year
# one can work with cusip6 since we have put together all different share classes and thus only have one company/share
# numbers of Cusip after 6 identify the share issue and a security check number thus one can proceed with CUSIP 7
# Get CUSIP 6 to link with linking table
ccm_complete['cusip 6'] = ccm_complete['cusip 9'].str[:6]


# In[30]:


ccm_complete.info()
ccm_complete.head()


# ## Transform CUSIPs into CIKs to link with dataframe containing CIKs and reporting dates
# 

# In[31]:


# CIKs given by Compustat are mostly correct however in instances where a company changed its CIK all of its prior
# CIK entries get overwritten by the new one and thus making it impossible to match the old statement with the CIK
# thus we use CUSIP (which is unique and does not change) to match textual analysis scorces with CIKs

# thus the CIKs provided by Compustat are partly misleading (at least the very old ones can be) and therefore 
# CIK will not be used to merge the dataframes until 2015 (more explanations below)


# ### Linking table from Github (https://github.com/volkovacodes/Block_Codes)

# In[32]:


# data only available until 2015
# from 2015 we will use the CIKs provided by Compustat assuming that the probability of CIKs having been abandoned 
# and then re-used for another company are marginal


# # Linkingtable

# In[33]:


os.chdir(PATH + r'/Linking table')
df_CUSIP_CIK = pd.read_csv('CIK_CUSIP.csv', dtype = {'CIK': object}) 
# data from prior to 2000 is in the table which is not needed for the linking process 
df_CUSIP_CIK = df_CUSIP_CIK.drop(df_CUSIP_CIK[df_CUSIP_CIK.Year <2000].index)
# rename column names to make them compatible with the ccm_complete column names for the merge
df_CUSIP_CIK = df_CUSIP_CIK.rename(columns={'Year': 'year', 'CUSIP6': 'cusip 6'})
df_CUSIP_CIK.info()
df_CUSIP_CIK.head()


# #### Testing purposes

# In[34]:


df_CUSIP_CIK.loc[df_CUSIP_CIK['cusip 6'] == '367204']


# In[35]:


merged_CRSP_Compustat_till15 = pd.merge(ccm_complete, df_CUSIP_CIK, on=['cusip 6', 'year'], how='inner')


# In[36]:


# 2 high-level checks on whether it has been merged correctly and whether the linking table is correct
# 1. SEC_Name (from linking table) agrees with comnam from CRSP
# 2. CIK (from linking table) agrees with CIK from CRSP

merged_CRSP_Compustat_till15.info()
merged_CRSP_Compustat_till15.head()


# #### Table has cleaned data until 2015   (because the linking table only had data until December 2015)

# In[37]:


# clean the table and delete rows judged to be of no more help
merged_CRSP_Compustat_till15.drop(columns = ['permco','comnam','cusip 9','year', 'cusip 6','cik','CUSIP','gvkey'])

merged_CRSP_Compustat_till15.head()


# #### Table from 2016 onwards ( no linking table will be used but the CIK's given by CRSP will be used)

# In[38]:


# previously no errors could be found by comparing linked CIKs and CIKs given by CRSP using .sample() multiple times
# internet argues that CIKs can be faulty in earlier years because companies go bankrupt and the CIKs get re-used
# since now all CIKs are from 2016 and this mistake is minimised (but there is no 100% guarantee that there are no 
# mistakes in the CIKs CRSP provides

# usually one would use the WRDS linking table which links CUSIP 9 with CIK without a mistake
# HOWEVER for that database another subscription to WRDS is needed that Lancaster does not pay for/provides


# In[39]:


splitting_date = pd.Timestamp(2016, 1, 1)
# only select entries more recent than 31/12/2015
merged_CRSP_Compustat_16onwards =  ccm_complete[ccm_complete['yearend'] >= splitting_date]
# rename column names to make them compatible with the merged_CRSP_Compustat_till15 column names for the merge
merged_CRSP_Compustat_16onwards = merged_CRSP_Compustat_16onwards.rename(columns={'cik': 'CIK'})
# drop all the rows who do not have a CIK number given by CRSP (for whatever reason)
merged_CRSP_Compustat_16onwards.dropna(subset=['CIK'])

merged_CRSP_Compustat_16onwards.info()
merged_CRSP_Compustat_16onwards.head()


# In[40]:


# merge to get a dataframe with CIKs for the whole period
merged_CRSP_Compustat_complete = pd.concat([merged_CRSP_Compustat_till15, merged_CRSP_Compustat_16onwards], ignore_index=True, sort=False)
# rename to avoid any mistakes later on
merged_CRSP_Compustat_complete = merged_CRSP_Compustat_complete.rename(columns={'date': 'shr prc date', 'cumretx': 'cumretxfy', 'lcumretx': 'lcumretxfy'})
# clean and delte unnecessary columns
merged_CRSP_Compustat_complete = merged_CRSP_Compustat_complete.drop(columns = ['permco','comnam','cusip 9','year', 'cusip 6','cik','CUSIP','gvkey'])
# change CIK from integer type to string type for later merging purposes

merged_CRSP_Compustat_complete.info()
merged_CRSP_Compustat_complete.head()


# ## Now link dataframe with returns, prices with textual analysis scores (via CIK's)

# #### Clean CIK table to get rid of faulty rows (as noticed earlier when cleaning text data and downloading to pickles)

# In[59]:


# drop the empty rows (should have done earlier but only noticed it now - they explain the empty pickle downloads)
data2000_2018C = data2000_2018C.dropna()
# indexes have to stay since they alow to link to the pickles (pickles have the name of the row index)

data2000_2018C.info()
data2000_2018C.head()

data2000_2018C['index'] = data2000_2018C.index


# #### For a perfect match between filing date and share price data date

# In[60]:


# Analysis_dataframe = pd.merge(data2000_2018C, merged_CRSP_Compustat_complete, how='inner',left_index=True , left_on=['Filing_date','CIK'], right_on=['shr prc date','CIK'])
Analysis_dataframe = pd.merge(data2000_2018C, merged_CRSP_Compustat_complete, how='inner' , left_on=['Filing_date','CIK'], right_on=['shr prc date','CIK'])


# In[64]:


# drop some more columns
Analysis_dataframe = Analysis_dataframe.drop(columns = ['yearend','datadate','jdate_x','jdate_y'])


# In[65]:


# share price and CIK verified with yahoo finance -> merging has been done in the correct way

Analysis_dataframe.info()
Analysis_dataframe.head()


# # output pickles

# In[1101]:


os.chdir(PATH + r'/Output pickles')


# In[2]:


# open a file, where you want to store the data
file = open('Dataframe_without_textualanalysis_scores', 'wb')

# dump information to that file
pickle.dump(Analysis_dataframe, file)

# close the file
file.close()


# In[221]:


infile = open('Dataframe_without_textualanalysis_scores','rb')
Analysis_dataframe = pickle.load(infile)
infile.close()


# In[222]:


Analysis_dataframe.sample(5)


# # Cleaning text data

# In[1081]:


#Function to clean the dataset
def url_to_clean_text_round_1(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()
    text = soup.get_text()    
    
    
    # convert to lower case
    text = text.lower()
    text = re.sub(r'(\t|\v)', '', text)
    # remove \xa0 which is non-breaking space from ISO 8859-1
    text = re.sub(r'\xa0', ' ', text)
    # remove newline feeds (\n) following hyphens
    text = re.sub(r'(-+)\n{2,}', r'\1', text)
    # remove hyphens preceded and followed by a blank space
    text = re.sub(r'\s-\s', '', text)
    # replace 'and/or' with 'and or'
    text = re.sub(r'and/or', r'and or', text)
    # tow or more hypens, periods, or equal signs, possiblly followed by spaces are removed
    text = re.sub(r'[-|\.|=]{2,}\s*', r'', text)
    # all underscores are removed
    text = re.sub(r'_', '', text)
    # 3 or more spaces are replaced by a single space
    text = re.sub(r'\s{3,}', ' ', text)
    # three or more line feeds, possibly separated by spaces are replaced by two line feeds
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    # remove hyphens before a line feed
    text = re.sub(r'-+\n', '\n', text)
    # replace hyphens preceding a capitalized letter with a space
    text = re.sub(r'-+([A-Z].*)', r' \1', text)
    # remove capitalized or all capitals for the months
    text = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)', '', text)
    # remove years
    text = re.sub(r'2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019', '', text)
    # remove words million and company 
    text = re.sub(r'million|company', '', text)  
    # remove line feeds
    text = re.sub('\n', ' ', text)
    #replace single line feed \n with single space
    text = re.sub(r'\n', ' ', text)
    return text


# In[224]:


# Tokenizeing module and filtering tokens using stop words list, removing punctuations
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words


# ## Stopwords list for cleaning text data

# In[1108]:


#StopWords from Loughran and McDonald - https://sraf.nd.edu/textual-analysis/resources/
os.chdir(PATH + r'/Dictionaries')

stopWordsFile = r'StopWords_Generic.txt'
#Loading stop words dictionary for removing stop words
with open(stopWordsFile ,'r') as stop_words:
    stopWords = stop_words.read().lower()
stopWordList = stopWords.split('\n')
stopWordList[-1:] = []


# In[226]:


stopWordList


# # Dictionnaries

# ## Defining wordlists for textual analysis (Loughran & McDonald)

# In[1071]:


#Import the negative and positive dictionaries 
##### Words from Loughran and McDonald
# negative 
neg_dict_LM = ""
neg_dict_LM = pd.read_csv(r'lm_negative.csv',encoding = 'ISO-8859-1', names=['lm_negative'])['lm_negative'].values.tolist()
neg_dict_LM = str(neg_dict_LM)
neg_dict_LM = neg_dict_LM.lower()

# positive
pos_dict_LM = ""
pos_dict_LM = pd.read_csv(r'lm_positive.csv', encoding = 'ISO-8859-1', names=['lm_positive'])['lm_positive'].values.tolist()
pos_dict_LM = str(pos_dict_LM)
pos_dict_LM = pos_dict_LM.lower()



# ## Defining wordlists for textual analysis (Bing - alternative to LM)

# In[1072]:



##### Words from Bing Liu
# negative
neg_dict_B = ""
neg_dict_B = pd.read_csv(r'bl_negative.csv', encoding = 'ISO-8859-1', names=['bl_negative'])['bl_negative'].values.tolist()
neg_dict_B = str(neg_dict_B)
neg_dict_B = neg_dict_B.lower()


# positive
pos_dict_B = ""
pos_dict_B = pd.read_csv(r'bl_positive.csv',encoding = 'ISO-8859-1', names=['bl_positive'])['bl_positive'].values.tolist()
pos_dict_B = str(pos_dict_B)
pos_dict_B = pos_dict_B.lower()


# ## ESG wordlist (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3206751)

# In[1073]:


# 422 words from dictionary imported from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3206751

ESG_dict = ['align', 'aligned','aligning','alignment','aligns','bylaw','bylaws','charter','charters','culture','death','duly','parents','independent','compliance','conduct','conformity','governance','misconduct','parachute','parachutes','perquisites','plane','planes','poison','retirement','approval','approvals','approve','approved','approves','approving','assess','assessed','assesses','assessing','assessment','assessments','audit','audited','auditing','auditor','auditors','audits','control','controls','coso','detect','detected','detecting','detection','evaluate','evaluated','evaluates','evaluating','evaluation','evaluations','examination','examinations','examine','examined','examines','examining','irs','oversee','overseeing','oversees','oversight','review','reviewed','reviewing','reviews','rotation','test','tested','testing','tests','treadway','backgrounds','independence','leadership','nomination','nominations','nominee','nominees','perspectives','qualifications','refreshment','skill','skills','succession','tenure','vacancies','vacancy','appreciation','award','awarded','awarding','awards','bonus','bonuses','cd','compensate','compensated','compensates','compensating','compensation','eip','iso','isos','payout','payouts','pension','prsu','prsus','recoupment','remuneration','reward','rewarding','rewards','rsu','rsus','salaries','salary','severance','vest','vested','vesting','vests','cast','consent','elect','elected','electing','election','elections','elects','ballot','ballots','cast','consent','nominate','nominated','plurality','proponent','proponents','proposal','proposals','proxies','quorum','vote','voted','votes','voting','attract','attracting','attracts','incentive','incentives','interview','interviews','motivate','motivated','motivates','motivating','motivation','recruit','recruiting','recruitment','retain','retainer','retainers','retaining','retention','talent','talented','talents','cobc','ethic','ethical','ethically','ethics','honesty','bribery','corrupt','corruption','crimes','whistleblower','embezzlement','grassroots','influence','influences','influencing','lobbied','lobbies','lobby','lobbying','lobbyist','lobbyists','announce','announced','announcement','announcements','announces','announcing','communicate','communicated','communicates','communicating','erm','fairly','integrity','liaison','presentation','presentations','asc','disclose','disclosed','discloses','disclosing','disclosure','disclosures','fasb','gaap','objectivity','press','sarbanes','sustainable,engagement','engagements','feedback','hotline','peace','investor','invite','invited','mail','mailed','mailing','mailings','notice','relations','bisexual','diversity','ethnic','ethnically','ethnicities','ethnicity','female','females','gay','gays','gender','genders','homosexual','immigration','lesbian','lesbians','lgbt','minorities','minority','ms','race','racial','religion','religious','sex','transgender','woman','women','stakeholder','stakeholders','charitable','charities','charity','donate','donated','donates','donating','donation','donations','donors','foundation','foundations','gift','gifts','nonprofit','povertycompact','ungc','communities','community','expression','marriage','privacy','occupational','safe','safely','safety','ilo','labour','eicc','endowment','endowments','people','philanthropic','philanthropy','socially','societal','society','welfare','biofuels','biofuel','childbirth','drug','medicaid','medicare','medicine','medicines','bargaining','eeo','fairness','fla','harassment','injury','labor','overtime','ruggie','sick','wage','wages','workplacegreen','renewable','solar','stewardship','wind','alcohol','drinking','disabled','human','nations','social','un','veteran','veterans','vulnerable','clean','environmental','epa','sustainability','climate','warming','dignity','discriminate','discriminated','discriminating','discrimination','equality','freedom','humanity','nondiscrimination','sexual','emission','emissions','ghg','ghgs','greenhouse','atmosphere','emit','agriculture','deforestation','bugs','conformance','defects','fda','inspection','inspections','minerals','standardization','warranty','pesticide','pesticides','wetlands','zoning','hiv','biphenyls','hazardous','householding','pollutants','printing','recycling','toxic','waste','wastes','weee','recycle','biodiversity','species','wilderness','employ','emp','wildlife','freshwater','groundwater','water','cleaner','cleanup','coal','contamination','fossil','resource','air','carbon','nitrogen','pollution','superfund','children','epidemic','health','healthy','ill','illness','pandemic','teachers','teaching','training']


# ## Textual analysis defintions

# In[1074]:


# avergae words per sentence 
def average_sentence_length(text):
    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
    
    return round(average_sent_length)

# Function to count the words
def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)

#Based on the dictionary of Loughran and McDonald (2016)
# Calculating positive score 
def positive_word_LM(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_LM:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos

# Calculating Negative score
def negative_word_LM(text):
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_LM:
            numNegWords -=1
    sumNeg = numNegWords 
    
    sumNeg = sumNeg * -1
    return sumNeg

#Based on the dictionary of Bing Liu 
# Calculating positive score 
def positive_word_B(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_B:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos

# Calculating Negative score
def negative_word_B(text):
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_B:
            numNegWords -=1
    sumNeg = numNegWords 
    sumNeg = sumNeg * -1
    return sumNeg

# Calculating polarity score
def polarity_score(positiveScore, negativeScore):
    pol_score = (positiveScore - negativeScore) / (positiveScore + negativeScore) 
    return pol_score

def ESG_percentage(text):
        numESGWords=0
        rawToken = tokenizer(text)
        for word in rawToken:
            if word in ESG_dict:
                numESGWords +=1
        
        sumESG = numESGWords 
        return sumESG

# Calculating Average sentence length 
# It will calculated using formula --- Average Sentence Length = the number of words / the number of sentences
     


# ### Preparation for textual analysis

# In[229]:


# transform from dataframe columns one numpy array containing only final links - quickest way when iterating is using numpy

### problem with 726,1357,1746,2013,2167,4252,4253,5423,5723,5885,6216,6394,6645,8762,9871,10331,10526,10840,11841,11842
### problem with 13471,14349,14810-21 and not downloaded apart from 1st one , 14929, 15811,15812,16788,17137,17773,20253
### problem with 26176,26329,28624,53051,62213, 2 mol, 102792, 105330, 105452, 107367, 107897, 115227, 123806,123974, 132413,
### problem with 134992,135310,140887,140905,143703
### all in all 55 empty files
Final_URLs = Analysis_dataframe['Form_url'].values.tolist() # probleem nummere + 1


# In[230]:


Final_URLs = np.asarray(Final_URLs)


# In[231]:


Final_URLs


# ## Not compulsory: Download cleaned financial statements for back-up

# In[1082]:


# function to simply gathering data

os.chdir(PATH + r'/Cleaned text')
    
for index, url in enumerate(Final_URLs,0):
    with open(str(index) + '.pkl', 'wb') as f:
        pickle.dump(url_to_clean_text_round_1(url), f)    


# In[1125]:


### Have a look at financial statement text data

Finstatnumb = 'YOUR NUMBER HERE'

financial_statement = open(Finstatnumb + '.pkl','rb')
new_dict = pickle.load(financial_statement)
financial_statement.close()


# # Textual analysis scores

# ## Not compulsory: Ends here

# In[23]:


# to have the layout in mind again
Analysis_dataframe.sample(5)


# In[1188]:


def textual_analysis(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()
    text = soup.get_text()    
    
    
    # convert to lower case
    text = text.lower()
    text = re.sub(r'(\t|\v)', '', text)
    # remove \xa0 which is non-breaking space from ISO 8859-1
    text = re.sub(r'\xa0', ' ', text)
    # remove newline feeds (\n) following hyphens
    text = re.sub(r'(-+)\n{2,}', r'\1', text)
    # remove hyphens preceded and followed by a blank space
    text = re.sub(r'\s-\s', '', text)
    # replace 'and/or' with 'and or'
    text = re.sub(r'and/or', r'and or', text)
    # tow or more hypens, periods, or equal signs, possiblly followed by spaces are removed
    text = re.sub(r'[-|\.|=]{2,}\s*', r'', text)
    # all underscores are removed
    text = re.sub(r'_', '', text)
    # 3 or more spaces are replaced by a single space
    text = re.sub(r'\s{3,}', ' ', text)
    # three or more line feeds, possibly separated by spaces are replaced by two line feeds
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    # remove hyphens before a line feed
    text = re.sub(r'-+\n', '\n', text)
    # replace hyphens preceding a capitalized letter with a space
    text = re.sub(r'-+([A-Z].*)', r' \1', text)
    # remove capitalized or all capitals for the months
    text = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)', '', text)
    # remove years
    text = re.sub(r'2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019', '', text)
    # remove words million and company 
    text = re.sub(r'million|company', '', text)  
    # remove line feeds
    text = re.sub('\n', ' ', text)
    #replace single line feed \n with single space
    text = re.sub(r'\n', ' ', text)
 

    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
        

# count the words
    tokens = tokenizer(text)
    
# count the positive words  
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_LM:
            numPosWords  += 1
    
    sumPos = numPosWords
    
# Calculating Negative score
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_LM:
            numNegWords -=1
    sumNeg = numNegWords 
    
    sumNeg = sumNeg * -1  

# Based on the dictionary of Bing Liu 
# Calculating positive score 
    numPosWordsB = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_B:
            numPosWordsB  += 1
    
    sumPosB = numPosWordsB 

# Based on the dictionary of Bing Liu 
# Calculating Negative score

    numNegWordsB=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_B:
            numNegWordsB -=1
    
    sumNegB = numNegWordsB 
    sumNegB = sumNegB * -1
    

# ESG words
    numESGWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in ESG_dict:
            numESGWords +=1
        
    sumESG = numESGWords 
        
    return round(average_sent_length), len(tokens),sumPos, sumNeg, sumPosB, sumNegB, sumESG, url


# In[30]:


os.chdir(PATH + r'/Textual analysis scores')


# ## Doing textual analysis and saving results in individual pickles

# In[222]:


for index, url in enumerate(Final_URLs):
    with open(str(index)+'.pkl','wb') as file:
        pickle.dump(textual_analysis(url), file)

# problems with 532, 24076, 25718, 26514, 40514 who are empty even after retrying a few times    
    


# ### Take care of 5 errors from above (NEED TO BE DONE MANUALLY 5 TIMES)

# In[ ]:


df = pd.DataFrame({-1:[0],
                   -11:[1],
                   -111:[2],
                   -1111:[3],
                   -11111:[4],
                   -111111:[5],
                   -1111111:[6],
                   'url':['no url']})


# In[ ]:


# doing this so that we can load all pickles back into a big dataframe and do not get any empty rows but instead rows
# with missing data contain the above content (easy recognisable)

outfile = open('NUMBER OF FILE WHICH HAD AN ERROR.pkl','wb')
pickle.dump(df,outfile)
outfile.close()


# ## Load textual analysis pickles back into big dataframe

# In[548]:


data = {}

for index,_ in enumerate(Final_URLs): # need to change the 100
    with open(str(index) + ".pkl", "rb") as f:
        data[index] = pickle.load(f)


# In[549]:


# random testing number to see output
data.get(22231)


# In[233]:


# transform into pandas dataframe
Textual_analysis_results = pd.DataFrame.from_dict(data,orient = 'index',columns = ['avg_sentence_length','nr_of_tokens','sumPosLM','sumNegLM','sumPosB','sumNegB','sumESG','url'])


# ## Define textual analysis metrics

# In[551]:


def polarity_score(positiveScore, negativeScore):
    pol_score = (positiveScore - negativeScore) / (positiveScore + negativeScore)
    return pol_score


# In[552]:


def ESG_percentage(sumESG,nr_of_tokens):
    ESG_perc = sumESG / nr_of_tokens
    return ESG_perc


# In[553]:


def NegB_percentage(sumNegB,nr_of_tokens):
    negB_perc = sumNegB / nr_of_tokens
    return negB_perc


# In[554]:


def NegLM_percentage(sumNegLM,nr_of_tokens):
    negLM_perc = sumNegLM / nr_of_tokens
    return negLM_perc


# In[555]:


def PosB_percentage(sumPosB,nr_of_tokens):
    PosB_perc = sumPosB / nr_of_tokens
    return PosB_perc


# In[556]:


def PosLM_percentage(sumPosLM,nr_of_tokens):
    PosLM_perc = sumPosLM / nr_of_tokens
    return PosLM_perc


# ## Calculate scores  and metrics

# In[557]:


Textual_analysis_results['polarity_LM'] = np.vectorize(polarity_score)(Textual_analysis_results['sumPosLM'],Textual_analysis_results['sumNegLM'])
Textual_analysis_results['polarity_B'] = np.vectorize(polarity_score)(Textual_analysis_results['sumPosB'],Textual_analysis_results['sumNegB'])
Textual_analysis_results['ESGpercentage'] = np.vectorize(ESG_percentage)(Textual_analysis_results['sumESG'],Textual_analysis_results['nr_of_tokens'])
Textual_analysis_results['NegBpercentage'] = np.vectorize(NegB_percentage)(Textual_analysis_results['sumNegB'],Textual_analysis_results['nr_of_tokens'])
Textual_analysis_results['NegLMpercentage'] = np.vectorize(NegLM_percentage)(Textual_analysis_results['sumNegLM'],Textual_analysis_results['nr_of_tokens'])
Textual_analysis_results['PosBpercentage'] = np.vectorize(PosB_percentage)(Textual_analysis_results['sumPosB'],Textual_analysis_results['nr_of_tokens'])
Textual_analysis_results['PosLMpercentage'] = np.vectorize(PosLM_percentage)(Textual_analysis_results['sumPosLM'],Textual_analysis_results['nr_of_tokens'])
Textual_analysis_results['Overall_sentiment_B'] = Textual_analysis_results['PosBpercentage'] - Textual_analysis_results['NegBpercentage']
Textual_analysis_results['Overall_sentiment_LM'] = Textual_analysis_results['PosLMpercentage'] - Textual_analysis_results['NegLMpercentage']


# In[558]:


Textual_analysis_results.head()


# ## Merge textual analysis scores with CRSP data (Analysis dataframe)

# In[559]:


# merge
Intermediatedf = pd.merge(Analysis_dataframe, Textual_analysis_results, how='inner' , left_on=['Form_url'], right_on=['url'])
# drop duplicates (should have done earlier (arised at the very beginning of this work but forgot to drop them at that point))
Textual_analysis_and_returns = Intermediatedf.drop_duplicates()


# In[560]:


Textual_analysis_and_returns.info()
Textual_analysis_and_returns.sample(5)


# #  Import NAICS linking table (last updated in 2017)

# In[1151]:


os.chdir(PATH + r'/Linking table')
# import
df_naics_linkingtable = pd.read_excel('2017_NAICS_Structure.xlsx',dtype = str)


# In[1152]:


# overview
df_naics_linkingtable.head()


# In[563]:


No_indsutry_indication = pd.DataFrame({'naics': [-1],
                   '2017 NAICS Title': ['no information available']})


# In[564]:


# dataframe contains now all data from above and one extra row for observations which do not match with the naics
df_naics_linkingtable = pd.concat([df_naics_linkingtable,No_indsutry_indication ], ignore_index=True, sort=False)


# In[565]:


# merge
Analysis = pd.merge(Textual_analysis_and_returns,df_naics_linkingtable,how = 'left',on = 'naics')


# In[566]:


# overview
Analysis.info()
Analysis.head()


# In[1153]:


# saving
os.chdir(PATH + r'/Analysis')

outfile = open('Analysis.pkl','wb')
pickle.dump(Analysis,outfile)
outfile.close()


### Analysis of data

