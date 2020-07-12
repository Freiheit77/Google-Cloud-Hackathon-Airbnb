# Google Cloud Hackathon: Airbnb
This project aims to use Google Cloud Platform to analyze the challenges faced by Airbnb as well as other trip/housing related companies during the coronavirus pandemic, provide robust solutions for survival, and give suggestions for post covid-19 development strategy. The impact of recommendations are evaluated at the end.

## Business Case


## Dataset
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

Source: http://insideairbnb.com/get-the-data.html

## Analytical Techniques: GCP
1. [BigQuery](https://github.com/Freiheit77/Google-Cloud-Hackathon-Airbnb/blob/master/docs/bigquery.png)

2. [Data Studio](https://github.com/Freiheit77/Google-Cloud-Hackathon-Airbnb/blob/master/docs/beijing.png)

3. [AI Platform](https://github.com/Freiheit77/Google-Cloud-Hackathon-Airbnb/blob/master/docs/AI%20platform.png)

## Model Develpment
1. [Interactive visualization for topics model of Beijing Airbnb](https://freiheit77.github.io/Google-Cloud-Hackathon-Airbnb/beijing_vis.html)

## Recommendations


## Impact


Reference:
https://github.com/Tony607/Chinese_sentiment_analysis/blob/master/chinese_sentiment_analysis.ipynb
