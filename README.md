# Google Cloud Hackathon: Airbnb
# Problem Overview
Faced with the increasingly serious situation and more and more confirmed cases, countries around the world have issued shelter-in-place policies, travel and cross-border tourism has been greatly reduced. 
This project aims to use Google Cloud Platform to analyze the challenges faced by Airbnb as well as other trip/housing related companies during the coronavirus pandemic, provide robust solutions for survival, and give suggestions for post covid-19 development strategy. The impact of recommendations are evaluated at the end.

## Data Gathering
We collected datasets from http://insideairbnb.com/get-the-data.html. The details of the datasets are shown below:

## Datasets
1. Listings (Beijing, San Francisco)
   - id: listing id
   - host_id: host id
   - name: listing name 
   - summary: the overall summary for the listings
   - notes: some special issues to be mentioned
   - transit: location and transportation
   - house_rules
   - picture_url
   - host_name
   - host_since
   - host_location
   - host_response_time
   - host_response_rate
   - host_acceptance_rate
   - host_listings_count
   - host_verifications
   - host_has_profile_pic
   - street
   - neighbourhood
   - city
   - state
   - zipcode
   - country_code
   - latitude
   - longitude
   - property_type
   - room_type
   - accommodates
   - bathrooms
   - bedrooms
   - beds
   - bed_type
   - amenities
   - price
   - weekly_price
   - monthly_price
   - security_deposit
   - cleaning_fee
   - guests_included
   - extrac_people
   - minimum_nights
   - maximum_nights
   - number of reviews
   - first_review
   - last_review
   - review_scores_rating
   - review_scores_accuracy
   - review_scores_cleanliness
   - review_scores_checkin
   - review_scores_communication
   - review_score_location
   - license
   - cancellation_policy
   - etc.
   
2. Reviews 
   - listing_id
   - id: review doc id
   - date: date of giving reviews
   - reviewer_id
   - reviewer_name
   - comments

## Storage Option: Google Cloud Storage
Since the overall size of all datasets accumulated to 1 GB + and number of features was over a hundred, we imported datasets to Google cloud storage.

## GCP Solution Chosen and Why (**Click on the link to look at the screenshots**)
We created a project on GCP so that all steps of data analytics are integrated on a single platform: 
1. [Storage](https://github.com/boyasun/Google-Cloud-Hackathon-Airbnb/blob/master/docs/storage.png)
Google Accounts include 15 GB of storage for free. Also, Storage makes it easier to share and manage datasets across the team.

2. [BigQuery](https://github.com/Freiheit77/Google-Cloud-Hackathon-Airbnb/blob/master/docs/bigquery.png)
Now that our project is connected to the storage which contains the bucket of datasets that we need, we can pull the data in the format that we want with just a few lines of SQL code on the BigQuery console. 

3. [Data Studio](https://github.com/Freiheit77/Google-Cloud-Hackathon-Airbnb/blob/master/docs/beijing.png)
Data Studio is directly connected with BigQuery. With just one click, we can use all kinds of plots and tables to explore insights from the data and create meaningful visualizations to guide further analysis.

4. [AI Platform](https://github.com/Freiheit77/Google-Cloud-Hackathon-Airbnb/blob/r/docs/AI%20platform.png)
We created Python3 Notebook Instances on the AI Platform to do further analysis including exploratory data analysis, topic modeling, etc. We also applied Google Cloud NLP API in our analysis. Overall speaking, GCP’s computational time and query efficiency are much higher than using some other techniques. 

## Model Performance
1. Showcase the visualization of our topic modeling: [Interactive visualization for topics model of Beijing Airbnb](https://freiheit77.github.io/Google-Cloud-Hackathon-Airbnb/beijing_vis.html)<br>
2. [Topic modeling coherence score](https://github.com/boyasun/Google-Cloud-Hackathon-Airbnb/blob/master/docs/modelperformance.png): The coherence score is for assessing the quality of the learned topics. Our models have coherence score around 0.5, which indicates a fairly good performance. 


## Recommendations&Impact
Based on the model results and our research, we have several recommendations for Airbnb to reopen their business in countries where lockdown restrictions has/are to lift. As it reopens its business in China, Airbnb can refer to its experience in China to set reopening strategies for other markets.

**Sanitation**: According to our topic model, sanitation and cleanliness is of the top concerns especially in the post virus phase. We recommend Airbnb to highlight safety and cleanliness score on their app or webpage for qualified hosts to provide assurance to guests. <br>
**Cooperation**: Airbnb’s cooperation with one of the popular video sharing app Kuaishou in China has successfully helped it promote new initiatives. We recommend Airbnb work with popular local platforms in different markets to attract hosts and guests. <br>
**Domestic Travel & Countryside Tour**: While Airbnb is already promoting domestic travel and countryside tour all over the world. Its experience in China indicates that these two strategies work and that partnering with local government departments of tourism can more efficiently promote the initiatives.   




Reference:
https://github.com/Tony607/Chinese_sentiment_analysis/blob/master/chinese_sentiment_analysis.ipynb
