
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

import re
import datetime as dt
from datetime import date
import os
import json 
import time

from selenium import webdriver 


# %%
# To fill
url = 'https://www.expedia.com/New-York-Hotels-Park-Central-Hotel-New-York.h4164.Hotel-Information?chkin=2021-02-24&chkout=2021-02-25&x_pwa=1&rfrr=HSR&pwa_ts=1607436721887&referrerUrl=aHR0cHM6Ly93d3cuZXhwZWRpYS5jb20vSG90ZWwtU2VhcmNo&useRewards=false&rm1=a2&regionId=178293&destination=New+York+%28and+vicinity%29%2C+New+York%2C+United+States+of+America&destType=MARKET&neighborhoodId=6141524&price=1&sort=RECOMMENDED&top_dp=98&top_cur=USD&semdtl=&selectedRoomType=202015972&selectedRatePlan=210276630'

hotel_name = 'Park Central Hotel New York'


# %%
# open driver
path = '/Users/camillebeguin/Downloads/chromedriver'
driver = webdriver.Chrome(path)
driver.get(url)


# %%
# open reviews
driver.find_element_by_css_selector('#app-layer-base > div > main > div.infosite__content.infosite__content--details > div.uitk-card-aloha-content-section.uitk-card-aloha-content-section-padded.uitk-card-aloha.uitk-card-aloha-roundcorner-all.uitk-spacing.uitk-flat-border-top.uitk-spacing-margin-blockend-three > div > div > div.uitk-layout-grid-item.uitk-layout-grid-item-columnspan-small-1.uitk-layout-grid-item-columnspan-medium-1.uitk-layout-grid-item-columnspan-large-7 > div.uitk-spacing.uitk-spacing-margin-small-blockstart-three.uitk-spacing-margin-medium-blockstart-two.uitk-spacing-padding-blockend-four.uitk-spacing-border-blockend > div > div.uitk-flex.uitk-flex-justify-content-space-between > button').click()    


# %%
# Click more reviews 
for i in range(1, 800):
    try:
        button = driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[2]/div/div/div/div[2]/section/div[2]/button')
        button.click()
    except:
        break


# %%
# Get html
html = driver.page_source
soup = BeautifulSoup(html)


# %%
# Store
hotel = {}

# Get hotel name and final score 
hotel['name'] = hotel_name
hotel['header'] = soup.find('div', 'all-b-margin-six')
hotel['average_score'] = hotel['header'].find('h1', {'class':'uitk-type-heading-600'}).text
hotel['count_reviews'] = hotel['header'].find('span', {'class': 'uitk-type-300'}).text
hotel['scores_amenities'] = hotel['header'].find_all('div', {'class': 'uitk-progress-bar-container'})

# Get reviews
reviews = soup.find_all('article', {'itemprop':'review'})


# %%
def extract_info_review(review):
    rating = review.find('span', {'itemprop': 'ratingValue'}).text
    author = review.find('span', {'itemprop': 'author'}).text
    traveller = review.find('div', {'class': 'uitk-type-300 uitk-text-secondary-theme'}).text
    date = review.find('span', {'itemprop': 'datePublished'}).text
    
    try:
        liked_disliked = review.find('section', {'data-stid': 'property-reviews-parent__section-sentiments'}).find_all('span', {'class': 'uitk-type-200 all-l-padding-two uitk-text-secondary-theme'})

        liked_disliked_1 = liked_disliked[0].text
        if liked_disliked_1.split(':')[0] == 'Liked':
            liked = liked_disliked_1.split(':')[1:]
            disliked = np.nan
        else:
            disliked = liked_disliked_1.split(':')[1:]
            liked = np.nan

        if len(liked_disliked) > 1:
            disliked = liked_disliked[1].text.split(':')[1:]
    except:
        liked, disliked = np.nan, np.nan
   
    try:
        description = review.find('span', {'itemprop': 'description'}).text
    except:
        description = np.nan
        
    return pd.DataFrame({
        'rating': [rating],
        'author': [author], 
        'travel_type': [traveller],
        'date': [date],
        'liked': [liked],
        'disliked': [disliked],
        'description': [description]
    })


def extract_info_reviews(reviews):
    all_reviews = pd.DataFrame()
    for review in reviews:
        all_reviews = pd.concat([all_reviews, extract_info_review(review)])
    return all_reviews


# %%
reviews_summary = extract_info_reviews(reviews).reset_index(drop=True)


# %%
def get_clean_reviews(reviews_df, hotel_name):
    
    # Replace dates in travel type
    reviews_df['travel_type'] = [np.nan if len(re.findall(r'\d+', x)) > 0 else x for x in reviews_df['travel_type']]
    
    # Format dates
    month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_number = list(range(1, 13))
    month_dic = dict(zip(month, month_number)) 
    
    reviews_df['day'] = [x.split(' ')[0] for x in reviews_df['date']]
    reviews_df['month'] = [x.split(' ')[1] for x in reviews_df['date']]
    reviews_df['year'] = [x.split(' ')[2] for x in reviews_df['date']]
    
    reviews_df['month'] = reviews_df['month'].replace(month_dic)
    reviews_df['date'] = reviews_df['day'].astype('str') + '-' + reviews_df['month'].astype('str') + '-' + reviews_df['year'].astype('str') 
    reviews_df['date'] = pd.to_datetime(reviews_df['date'], format='%d-%M-%Y')
    reviews_df.drop(columns=['day', 'month', 'year'], inplace=True)
    
    # Format rating
    reviews_df['rating'] = [x.split('/')[0] for x in reviews_df['rating']]
    
    # Add URL and hotel name
    reviews_df['url'] = url
    reviews_df['hotel_name'] = hotel_name
    
    # Return the dataframe
    return reviews_df


# %%
def get_clean_hotel_summary(hotel_dic):
    hotel_df = pd.DataFrame()
    
    # Format dictionary
    hotel_df = pd.DataFrame({
    'name': hotel_dic['name'],
    'average_rating': [float(hotel_dic['average_score'].split('/')[0])],
    'count_reviews': [int(hotel_dic['count_reviews'].split(' ')[0].replace(',', ''))],
    'rating_by_category': [[x.text for x in hotel_dic['scores_amenities']]],
    'cleanliness': [hotel_dic['scores_amenities'][0].text],
    'staff': [hotel_dic['scores_amenities'][1].text],
    'amenities': [hotel_dic['scores_amenities'][2].text],
    'property_facilities': [hotel_dic['scores_amenities'][3].text],
    'url': [url]
})
    return hotel_df


# %%
hotel_df = get_clean_hotel_summary(hotel)
reviews_df = get_clean_reviews(reviews_summary, hotel_name=hotel['name'])


# %%
hotel_df.head()


# %%
reviews_df.head()


# %%
print('Example of review: \n > %s \n > %s' % 
(reviews_df.loc[:, 'description'][0], reviews_df.loc[:, 'description'][100]))


# %%
hotel_df_all = pd.read_csv('expedia_hotels_multiple_brands.csv')
hotel_df_all = pd.concat([hotel_df_all, hotel_df])


# %%
filename = "_".join(['expedia', hotel_name, 'reviews.csv'])
reviews_df.to_csv(filename, index=False)
hotel_df_all.to_csv('expedia_hotels_multiple_brands.csv', index=False)


