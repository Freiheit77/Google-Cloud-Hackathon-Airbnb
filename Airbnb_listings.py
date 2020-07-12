#!/usr/bin/env python
# coding: utf-8

# # Import libraries and data sets

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[23]:


from google.cloud import storage
from google.cloud.storage.blob import Blob
import os


# In[24]:


client = storage.Client()
bucket = "hackathon0711"


# In[25]:


from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(os.getcwd()+"/"+destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


# In[26]:


download_blob("hackathon0711", "listings 2019April.csv.gz", "listings 2019April.csv.gz")
download_blob("hackathon0711", "listings 2019March.csv.gz", "listings 2019March.csv.gz")
download_blob("hackathon0711", "listings 2019May.csv.gz", "listings 2019May.csv.gz")
download_blob("hackathon0711", "listings 2020April.csv.gz", "listings 2020April.csv.gz")
download_blob("hackathon0711", "listings 2020March.csv.gz", "listings 2020March.csv.gz")
download_blob("hackathon0711", "listings 2020May.csv.gz", "listings 2020May.csv.gz")


# In[27]:


pd.set_option('display.max_columns', None)

df_sf_May_2020 = pd.read_csv("listings 2020May.csv.gz")
df_sf_May_2019 = pd.read_csv("listings 2019May.csv.gz")

df_sf_april_2020 = pd.read_csv("listings 2020April.csv.gz")
df_sf_april_2019 = pd.read_csv("listings 2019April.csv.gz")

df_sf_march_2020 = pd.read_csv("listings 2020March.csv.gz")
df_sf_march_2019 = pd.read_csv("listings 2019March.csv.gz")


# In[28]:


df_bj_May_2020 = pd.read_csv("bj_2020_may.csv.gz")
df_bj_May_2019 = pd.read_csv("bj_2019_may.csv.gz")

df_bj_april_2020 = pd.read_csv("bj_2020_apr.csv.gz")
df_bj_april_2019 = pd.read_csv("bj_2019_apr.csv.gz")

df_bj_march_2020 = pd.read_csv("bj_2020_mar.csv.gz")
df_bj_march_2019 = pd.read_csv("bj_2019_mar.csv.gz")


# In[29]:


review_sf = pd.read_csv("reviews_sf_Jun08.csv.gz")


# ## Analyze the change before and after the Covid-19 - San Francisco

# In[31]:


# index the data
df_sf_May_2020 = df_sf_May_2020.set_index(['id'])
df_sf_May_2019 = df_sf_May_2019.set_index(['id'])
df_sf_april_2020 = df_sf_april_2020.set_index(['id'])
df_sf_april_2019 = df_sf_april_2019.set_index(['id'])
df_sf_march_2020 = df_sf_march_2020.set_index(['id'])
df_sf_march_2019 = df_sf_march_2019.set_index(['id'])

# add tag for columns
df_sf_May_2020.columns = [str(col) + '_2020May' for col in df_sf_May_2020.columns]
df_sf_May_2019.columns = [str(col) + '_2019May' for col in df_sf_May_2019.columns]

df_sf_april_2020.columns = [str(col) + '_2020apr' for col in df_sf_april_2020.columns]
df_sf_april_2019.columns = [str(col) + '_2019apr' for col in df_sf_april_2019.columns]

df_sf_march_2020.columns = [str(col) + '_2020mar' for col in df_sf_march_2020.columns]
df_sf_march_2019.columns = [str(col) + '_2019mar' for col in df_sf_march_2019.columns]


# In[32]:


#Merge dataset in the same month of 2019 and 2020
ab_march = df_sf_march_2019.merge(df_sf_march_2020,left_index=True, right_index=True)
ab_may = df_sf_May_2019.merge(df_sf_May_2020,left_index=True, right_index=True)


# In[33]:


# clean the price march
ab_march['price_2019mar']= ab_march['price_2019mar'].str.replace("$","")
ab_march['price_2019mar'] = ab_march['price_2019mar'].str.replace(".",".")
ab_march['price_2019mar'] = ab_march['price_2019mar'].str.replace(",","")
ab_march['price_2019mar'] = pd.to_numeric(ab_march['price_2019mar'])

ab_march['weekly_price_2020mar']= ab_march['weekly_price_2020mar'].str.replace("$","")
ab_march['weekly_price_2020mar'] = ab_march['weekly_price_2020mar'].str.replace(".",".")
ab_march['weekly_price_2020mar'] = ab_march['weekly_price_2020mar'].str.replace(",","")
ab_march['weekly_price_2020mar'] = pd.to_numeric(ab_march['weekly_price_2020mar'])

ab_march['weekly_price_2019mar']= ab_march['weekly_price_2019mar'].str.replace("$","")
ab_march['weekly_price_2019mar'] = ab_march['weekly_price_2019mar'].str.replace(".",".")
ab_march['weekly_price_2019mar'] = ab_march['weekly_price_2019mar'].str.replace(",","")
ab_march['weekly_price_2019mar'] = pd.to_numeric(ab_march['weekly_price_2019mar'])

ab_march['monthly_price_2020mar']= ab_march['monthly_price_2020mar'].str.replace("$","")
ab_march['monthly_price_2020mar'] = ab_march['monthly_price_2020mar'].str.replace(".",".")
ab_march['monthly_price_2020mar'] = ab_march['monthly_price_2020mar'].str.replace(",","")
ab_march['monthly_price_2020mar'] = pd.to_numeric(ab_march['monthly_price_2020mar'])

ab_march['monthly_price_2019mar']= ab_march['monthly_price_2019mar'].str.replace("$","")
ab_march['monthly_price_2019mar'] = ab_march['monthly_price_2019mar'].str.replace(".",".")
ab_march['monthly_price_2019mar'] = ab_march['monthly_price_2019mar'].str.replace(",","")
ab_march['monthly_price_2019mar'] = pd.to_numeric(ab_march['monthly_price_2019mar'])


# In[34]:


# clean price for may
ab_may['price_2019May']= ab_may['price_2019May'].str.replace("$","")
ab_may['price_2019May'] = ab_may['price_2019May'].str.replace(".",".")
ab_may['price_2019May'] = ab_may['price_2019May'].str.replace(",","")
ab_may['price_2019May'] = pd.to_numeric(ab_may['price_2019May'])

ab_may['weekly_price_2020May']= ab_may['weekly_price_2020May'].str.replace("$","")
ab_may['weekly_price_2020May'] = ab_may['weekly_price_2020May'].str.replace(".",".")
ab_may['weekly_price_2020May'] = ab_may['weekly_price_2020May'].str.replace(",","")
ab_may['weekly_price_2020May'] = pd.to_numeric(ab_may['weekly_price_2020May'])

ab_may['weekly_price_2019May']= ab_may['weekly_price_2019May'].str.replace("$","")
ab_may['weekly_price_2019May'] = ab_may['weekly_price_2019May'].str.replace(".",".")
ab_may['weekly_price_2019May'] = ab_may['weekly_price_2019May'].str.replace(",","")
ab_may['weekly_price_2019May'] = pd.to_numeric(ab_may['weekly_price_2019May'])

ab_may['monthly_price_2020May']= ab_may['monthly_price_2020May'].str.replace("$","")
ab_may['monthly_price_2020May'] = ab_may['monthly_price_2020May'].str.replace(".",".")
ab_may['monthly_price_2020May'] = ab_may['monthly_price_2020May'].str.replace(",","")
ab_may['monthly_price_2020May'] = pd.to_numeric(ab_may['monthly_price_2020May'])

ab_may['monthly_price_2019May']= ab_may['monthly_price_2019May'].str.replace("$","")
ab_may['monthly_price_2019May'] = ab_may['monthly_price_2019May'].str.replace(".",".")
ab_may['monthly_price_2019May'] = ab_may['monthly_price_2019May'].str.replace(",","")
ab_may['monthly_price_2019May'] = pd.to_numeric(ab_may['monthly_price_2019May'])


# ### **Compare number of listings in the platform BEFORE and AFTER COVID19:**

# In[35]:


## listing number in sf in 2020 vs.2019
print("Listing number in SF in May,2020 is:{}".format(df_sf_May_2020.shape[0]))
print("Listing number in SF in May,2019 is:{}".format(df_sf_May_2019.shape[0]))
print("Listing number in SF in April,2020 is:{}".format(df_sf_april_2020.shape[0]))
print("Listing number in SF in April,2019 is:{}".format(df_sf_april_2019.shape[0]))
print("Listing number in SF in March,2020 is:{}".format(df_sf_march_2020.shape[0]))
print("Listing number in SF in March,2019 is:{}".format(df_sf_march_2019.shape[0]))


# There's not much change. The Number of Listing even increased after Covid19.
# But this is not all of the story. A lot of hosts closed the availability of the listing instead of remove the listing from the platform.

# ### **Compare number of listings close the booking availability for next month/season in the platform BEFORE and AFTER COVID19:**

# In[39]:


# display the availability_30 and availability_90 in table

avai_30 = [df_sf_May_2020[df_sf_May_2020['availability_30_2020May'] == 0].shape[0],
          df_sf_May_2019[df_sf_May_2019['availability_30_2019May'] == 0].shape[0],
          df_sf_april_2020[df_sf_april_2020['availability_30_2020apr'] == 0].shape[0],
         df_sf_april_2019[df_sf_april_2019['availability_30_2019apr'] == 0].shape[0],
       df_sf_march_2020[df_sf_march_2020['availability_30_2020mar'] == 0].shape[0],
       df_sf_march_2019[df_sf_march_2019['availability_30_2019mar'] == 0].shape[0] ]

avai_90 = [
df_sf_May_2020[df_sf_May_2020['availability_90_2020May'] == 0].shape[0],
df_sf_May_2019[df_sf_May_2019['availability_90_2019May'] == 0].shape[0],
df_sf_april_2020[df_sf_april_2020['availability_90_2020apr'] == 0].shape[0],
df_sf_april_2019[df_sf_april_2019['availability_90_2019apr'] == 0].shape[0],
df_sf_march_2020[df_sf_march_2020['availability_90_2020mar'] == 0].shape[0],
df_sf_march_2019[df_sf_march_2019['availability_90_2019mar'] == 0].shape[0],
]
year = [2020,2019,2020,2019,2020,2019]
month = ["May","May","April","April","March","March"]

avail = pd.DataFrame(list(zip(year, month, avai_30, avai_90)),
               columns =['Year', 'Month','N_Listing_without_availability_next_month',"N_Listing_without_availability_next_season"])
avail


# In[43]:


fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(20,8))
sns.barplot(x="Month", y="N_Listing_without_availability_next_month", hue="Year", data=avail,ax = axs[0] )
sns.barplot(x="Month", y="N_Listing_without_availability_next_month", hue="Year", data=avail,ax = axs[1] )


# **Compare price distribution of listings in SF under the impace of COVID19**

# Normally, when the economies is down, people usually give discounts to customers so that they make up certain proportion of revenue. Is this true for the hosts of Airbnb?

# In[13]:


# 2019 March vs. 2020 March
import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(20,20))
sns.violinplot(x=ab_march["weekly_price_2019mar"], ax=axs[0,0])
sns.violinplot(x=ab_march["weekly_price_2020mar"], ax=axs[0,1])
sns.violinplot(x=ab_march["monthly_price_2019mar"], ax=axs[1,0])
sns.violinplot(x=ab_march["monthly_price_2020mar"], ax=axs[1,1])
plt.show()


# **No. We found that there almost no change in the price distribution for both monthly rent and weekly rent.**

# In[14]:


# 2019 may vs. 2020 may
import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(20,20))
sns.violinplot(x=ab_may["weekly_price_2019May"], ax=axs[0,0])
sns.violinplot(x=ab_may["weekly_price_2020May"], ax=axs[0,1])
sns.violinplot(x=ab_may["monthly_price_2019May"], ax=axs[1,0])
sns.violinplot(x=ab_may["monthly_price_2020May"], ax=axs[1,1])
plt.show()


# **Compare the distribution of personal hosts number and Agency Hosts number**
# Usually, personal hosts are real 'hosts' that are advocated by Airbnb. They hold small amount of listing, usually less than 3 listings per people. Let's see how does the COVID 19 impact the personal and agency hosts.

# In[42]:


may2020_dist = ab_may.groupby("host_listings_count_2020May",as_index=False).count()
may2019_dist = ab_may.groupby("host_listings_count_2019May",as_index=False).count()


# In[46]:


may2020_dist.head()


# In[45]:


import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=1,nrows=2,figsize=(15,20))
sns.barplot(x="host_listings_count_2020May",y="listing_url_2019May",df= may2020_dist, ax=axs[0])
sns.barplot(x="host_listings_count_2019May",y="listing_url_2020May",df= may2020_dist, ax=axs[1])


# *这里这个一开始图画的不对 需要重新画*

# # ## Analyze the change before and after the Covid-19 - Beijing

# In[17]:


download_blob("hackathon0711", "bj_2020_may.csv.gz", "bj_2020_may.csv.gz")
download_blob("hackathon0711", "bj_2019_may.csv.gz", "bj_2019_may.csv.gz")
download_blob("hackathon0711", "bj_2020_apr.csv.gz", "bj_2020_apr.csv.gz")
download_blob("hackathon0711", "bj_2019_apr.csv.gz", "bj_2019_apr.csv.gz")
download_blob("hackathon0711", "bj_2020_mar.csv.gz", "bj_2020_mar.csv.gz")
download_blob("hackathon0711", "bj_2019_mar.csv.gz", "bj_2019_mar.csv.gz")


# In[47]:


df_bj_May_2020.head()


# In[48]:


df_bj_May_2020 = df_bj_May_2020.set_index(['id'])
df_bj_May_2019 = df_bj_May_2019.set_index(['id'])
df_bj_april_2020 = df_bj_april_2020.set_index(['id'])
df_bj_april_2019 = df_bj_april_2019.set_index(['id'])
df_bj_march_2020 = df_bj_march_2020.set_index(['id'])
df_bj_march_2019 = df_bj_march_2019.set_index(['id'])

df_bj_May_2020.columns = [str(col) + '_2020May' for col in df_bj_May_2020.columns]
df_bj_May_2019.columns = [str(col) + '_2019May' for col in df_bj_May_2019.columns]

df_bj_april_2020.columns = [str(col) + '_2020apr' for col in df_bj_april_2020.columns]
df_bj_april_2019.columns = [str(col) + '_2019apr' for col in df_bj_april_2019.columns]

df_bj_march_2020.columns = [str(col) + '_2020mar' for col in df_bj_march_2020.columns]
df_bj_march_2019.columns = [str(col) + '_2019mar' for col in df_bj_march_2019.columns]


# In[49]:


ab_march_bj = df_bj_march_2019.merge(df_bj_march_2020,left_index=True, right_index=True)
ab_may_bj = df_bj_May_2019.merge(df_bj_May_2020,left_index=True, right_index=True)


# In[50]:


# clean the price in airbnb
ab_march_bj['price_2019mar']= ab_march_bj['price_2019mar'].str.replace("$","")
ab_march_bj['price_2019mar'] = ab_march_bj['price_2019mar'].str.replace(".",".")
ab_march_bj['price_2019mar'] = ab_march_bj['price_2019mar'].str.replace(",","")
ab_march_bj['price_2019mar'] = pd.to_numeric(ab_march_bj['price_2019mar'])

ab_march_bj['weekly_price_2020mar']= ab_march_bj['weekly_price_2020mar'].str.replace("$","")
ab_march_bj['weekly_price_2020mar'] = ab_march_bj['weekly_price_2020mar'].str.replace(".",".")
ab_march_bj['weekly_price_2020mar'] = ab_march_bj['weekly_price_2020mar'].str.replace(",","")
ab_march_bj['weekly_price_2020mar'] = pd.to_numeric(ab_march_bj['weekly_price_2020mar'])

ab_march_bj['weekly_price_2019mar']= ab_march_bj['weekly_price_2019mar'].str.replace("$","")
ab_march_bj['weekly_price_2019mar'] = ab_march_bj['weekly_price_2019mar'].str.replace(".",".")
ab_march_bj['weekly_price_2019mar'] = ab_march_bj['weekly_price_2019mar'].str.replace(",","")
ab_march_bj['weekly_price_2019mar'] = pd.to_numeric(ab_march_bj['weekly_price_2019mar'])

ab_march_bj['monthly_price_2020mar']= ab_march_bj['monthly_price_2020mar'].str.replace("$","")
ab_march_bj['monthly_price_2020mar'] = ab_march_bj['monthly_price_2020mar'].str.replace(".",".")
ab_march_bj['monthly_price_2020mar'] = ab_march_bj['monthly_price_2020mar'].str.replace(",","")
ab_march_bj['monthly_price_2020mar'] = pd.to_numeric(ab_march_bj['monthly_price_2020mar'])

ab_march_bj['monthly_price_2019mar']= ab_march_bj['monthly_price_2019mar'].str.replace("$","")
ab_march_bj['monthly_price_2019mar'] = ab_march_bj['monthly_price_2019mar'].str.replace(".",".")
ab_march_bj['monthly_price_2019mar'] = ab_march_bj['monthly_price_2019mar'].str.replace(",","")
ab_march_bj['monthly_price_2019mar'] = pd.to_numeric(ab_march_bj['monthly_price_2019mar'])


# ### **Compare number of listings in the platform BEFORE and AFTER COVID19:**

# In[51]:


## listing number in bj in 2020 vs.2019
print("Listing number in BJ in May,2020 is:{}".format(df_bj_May_2020.shape[0]))
print("Listing number in BJ in May,2019 is:{}".format(df_bj_May_2019.shape[0]))
print("Listing number in bBJ in April,2020 is:{}".format(df_bj_april_2020.shape[0]))
print("Listing number in BJ in April,2019 is:{}".format(df_bj_april_2019.shape[0]))
print("Listing number in BJ in March,2020 is:{}".format(df_bj_march_2020.shape[0]))
print("Listing number in BJ in March,2019 is:{}".format(df_bj_march_2019.shape[0]))


# **Listing number in China has increased !**
#

# In[52]:


## high availability among listing?
print("30-day availability:")
print(df_bj_May_2020[df_bj_May_2020['availability_30_2020May'] == 0].shape[0])
print(df_bj_May_2019[df_bj_May_2019['availability_30_2019May'] == 0].shape[0])
print(df_bj_april_2020[df_bj_april_2020['availability_30_2020apr'] == 0].shape[0])
print(df_bj_april_2019[df_bj_april_2019['availability_30_2019apr'] == 0].shape[0])
print(df_bj_march_2020[df_bj_march_2020['availability_30_2020mar'] == 0].shape[0])
print(df_bj_march_2019[df_bj_march_2019['availability_30_2019mar'] == 0].shape[0])

print("90-day availability:")
print(df_bj_May_2020[df_bj_May_2020['availability_90_2020May'] == 0].shape[0])
print(df_bj_May_2019[df_bj_May_2019['availability_90_2019May'] == 0].shape[0])
print(df_bj_april_2020[df_bj_april_2020['availability_90_2020apr'] == 0].shape[0])
print(df_bj_april_2019[df_bj_april_2019['availability_90_2019apr'] == 0].shape[0])
print(df_bj_march_2020[df_bj_march_2020['availability_90_2020mar'] == 0].shape[0])
print(df_bj_march_2019[df_bj_march_2019['availability_90_2019mar'] == 0].shape[0])


# In[53]:


# display the availability_30 and availability_90 in table
total_listing = [df_bj_May_2020.shape[0],
df_bj_May_2019.shape[0],
df_bj_april_2020.shape[0],
df_bj_april_2019.shape[0],
df_bj_march_2020.shape[0],
df_bj_march_2019.shape[0]]

avai_30 = [df_bj_May_2020[df_bj_May_2020['availability_30_2020May'] == 0].shape[0],
          df_bj_May_2019[df_bj_May_2019['availability_30_2019May'] == 0].shape[0],
          df_bj_april_2020[df_bj_april_2020['availability_30_2020apr'] == 0].shape[0],
         df_bj_april_2019[df_bj_april_2019['availability_30_2019apr'] == 0].shape[0],
       df_bj_march_2020[df_bj_march_2020['availability_30_2020mar'] == 0].shape[0],
       df_bj_march_2019[df_bj_march_2019['availability_30_2019mar'] == 0].shape[0] ]

avai_90 = [
df_bj_May_2020[df_bj_May_2020['availability_90_2020May'] == 0].shape[0],
df_bj_May_2019[df_bj_May_2019['availability_90_2019May'] == 0].shape[0],
df_bj_april_2020[df_bj_april_2020['availability_90_2020apr'] == 0].shape[0],
df_bj_april_2019[df_bj_april_2019['availability_90_2019apr'] == 0].shape[0],
df_bj_march_2020[df_bj_march_2020['availability_90_2020mar'] == 0].shape[0],
df_bj_march_2019[df_bj_march_2019['availability_90_2019mar'] == 0].shape[0],
]

year = [2020,2019,2020,2019,2020,2019]

month = ["May","May","April","April","March","March"]

avail_bj = pd.DataFrame(list(zip(year, month, avai_30, avai_90, total_listing)),
               columns =['Year', 'Month','N_Listing_without_availability_next_month',"N_Listing_without_availability_next_season",
                         "No_total_listing"])
avail_bj


# ### Let's further explore what people care about when looking for a place to stay.

# In[22]:


df_bj_May_2020["review_scores_rating"] = df_bj_May_2020.apply(lambda x: float(x["review_scores_rating"]),axis=1)


# In[23]:


bj_review_score = df_bj_May_2020[["id","review_scores_cleanliness",
                            "review_scores_checkin","review_scores_communication",
                           "review_scores_location"]]
bj_review_score = bj_review_score.melt(id_vars=["id"],
        var_name="review_type",
        value_name="value")
bj_review_score.head(5)


# In[24]:


# visualization
import matplotlib.pyplot as plt
fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(20,14))
sns.violinplot(x=bj_review_score["review_type"],y=bj_review_score["value"])


# **The scores for clenliness is worthy notice, it's lower than the scores in other dimension(communication, checkin, location). It indicates that the cleanliness service needs extra attention if Airbnb want to retain more customers.**
