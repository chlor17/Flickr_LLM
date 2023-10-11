# Databricks notebook source
# %pip uninstall flickrapi
# %pip install flickrapi
%pip install git+https://github.com/beaufour/flickrapi.git
dbutils.library.restartPython()

# COMMAND ----------

from flickrapi import FlickrAPI


# COMMAND ----------

# MAGIC %pip freeze 

# COMMAND ----------

class Flickr_conn:
    def __init__(self, key=None, secret=None):
        if key:
            self.key = key
        else:
            self.key = config_file['FLICKR']['API_KEY']
        if secret:
            self.secret = secret
        else:
            self.secret = config_file['FLICKR']['SECRET']

        self.conn = None

    def create_connection(self):
      self.conn = FlickrAPI(self.key, self.secret)

# COMMAND ----------

conn = FlickrAPI("462f8e29852bb3330a49734ae6cf1d59", "ecc275b45d62e864")
SIZES = ["url_o", "url_k", "url_h", "url_l", "url_c"]  # in order of preference
extras = ','.join(SIZES)

a = conn.walk(text="montreal",  # it will search by image title and image tags
            extras=extras,  # get the urls for each size we want
            privacy_filter=1,  # search only for public photos
            per_page=50,
            sort='relevance')
# a = conn.walk(text="montreal",
#             user_id="131401388@N03",
#             extras= ["url_h"],
#             privacy_filter=1, 
#             per_page=2,
#             sort='relevance')
print(a)
for i in a:
  print(i)

# COMMAND ----------

from PIL import Image
import requests
import os
import sys
import time

class Photos:
    def __init__(self, image_name, user_id, destination_path = "/Volumes/chlor_catalog/flickr_llm/photos/"):
        self.image_name = image_name
        self.user_id = user_id
        self.destination_path = destination_path
        self.image_path = os.path.join(self.destination_path, self.image_name)
        self.photo_size = ["url_o", "url_k", "url_h", "url_l", "url_c"]
        self.url = None


    def get_url(self, conn): # should be only 1, assumes all image name are different
        photos_list = conn.walk(text=self.image_name,
                              user_id=self.user_id,
                              extras= ','.join(self.photo_size),
                              privacy_filter=1, 
                              per_page=50,
                              sort='relevance')
        print(photos_list)
        # try:
        for photo in photos_list:
            for size in range(len(self.photo_size)):  # makes sure the loop is done in the order we want
                self.url = photo.get(self.photo_size[size])
        # except:
        #     print("Photo list seems empty")

    def download_image(self):
        # if not os.path.isdir(self.destination_path):
        #     os.makedirs(self.destination_path)
        try:
            image_name = self.url.split("/")[-1]
            if not os.path.isfile(self.image_path):  # ignore if already downloaded
              response=requests.get(self.url,stream=True)

              with open(self.image_path,'wb') as outfile:
                  outfile.write(response.content)
        except:
            print("URL is empty")

    def print_image(self):
      image = Image.open(os.path.join(self.destination_path, self.image_name)) 
      print(image)


# COMMAND ----------

KEY = '462f8e29852bb3330a49734ae6cf1d59'
SECRET = 'ecc275b4562e864' # 'ecc275b45d62e864'

USER = "131401388@N03"
IMAGE_NAME = "2023_Montreal_271"

f_conn = Flickr_conn(key=KEY, secret=SECRET)
f_conn.create_connection()
print(f_conn.conn)
photo_1 = Photos(image_name = IMAGE_NAME, user_id = USER)

photo_1.get_url(f_conn.conn)
print("url : ", photo_1.url)
photo_1.download_image()



# COMMAND ----------

photo_1.url

# COMMAND ----------



# COMMAND ----------

from PIL import Image

image = Image.open('/Volumes/chlor_catalog/flickr_llm/photos/2023_Teracea_236') 
image

# COMMAND ----------



# COMMAND ----------

f_conn.create_connection()

p = f_conn.conn.walk(text="2023_Montreal_188",
                     user_id="131401388@N03",
                     extras= ','.join(["url_o", "url_k", "url_h", "url_l", "url_c"]),
                     privacy_filter=1, 
                     per_page=1,
                     sort='relevance')

# COMMAND ----------

photo_1.image_path

# COMMAND ----------

from PIL import Image

image = Image.open('/Volumes/chlor_catalog/flickr_llm/photos/2023_Montreal_222.jpg') 
image

# COMMAND ----------

from PIL import Image

flickr_api.set_keys(api_key = FLICKR_API_KEYS, api_secret = FLICKR_SECRETS)


# COMMAND ----------

response=requests.get("https://www.flickr.com/photos/lortie.chad/33137784286/",stream=True)
response.content

# COMMAND ----------

with open("/Volumes/chlor_catalog/flickr_llm/photos/00001.jpg",'wb') as outfile:
                outfile.write(response.content)

# COMMAND ----------

# Resize the image and overwrite it
image = Image.open('/Volumes/chlor_catalog/flickr_llm/photos/00001.jpg') 
image = image.resize((256, 256), Image.ANTIALIAS)

# COMMAND ----------

user = flickr_api.Person.findByUserName("lortie.chad")
photos = user.getPhotos()

# COMMAND ----------

photos[0]

# COMMAND ----------

user.getPhotos(photos[0]["id"])

# COMMAND ----------

flickr_api.person.getPhotos(photos[0]["id"])


# COMMAND ----------


# Download image from the url and save it to '00001.jpg'
urllib.request.urlretrieve("https://www.flickr.com/photos/131401388@N03/33137784286/", '/Volumes/chlor_catalog/flickr_llm/photos/00001.jpg')

# Resize the image and overwrite it
image = Image.open('/Volumes/chlor_catalog/flickr_llm/photos/00001.jpg') 
image = image.resize((256, 256), Image.ANTIALIAS)

# COMMAND ----------

import flickrapi
import urllib
import os

# Set your Flickr API key and secret
API_KEY = FLICKR_API_KEYS
API_SECRET = FLICKR_SECRETS

# Initialize the Flickr API client
flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, format='parsed-json')

def download_flickr_image(photo_id, save_path):
    try:
        # Get the photo info
        photo_info = flickr.photos.getInfo(photo_id=photo_id)
        photo_url = photo_info['photo']['urls']['url'][0]['_content']

        # Download the image
        urllib.request.urlretrieve(photo_url, save_path)
        print(f"Image downloaded successfully and saved as {save_path}")
    except Exception as e:
        print(f"Error downloading image: {e}")

flickr_photo_id = "53203960456"  # Replace with the Flickr photo ID you want to download
save_path = '/Volumes/chlor_catalog/flickr_llm/photos/00001.jpg'  # Specify the path where you want to save the image

download_flickr_image(flickr_photo_id, save_path)


# COMMAND ----------


from requests_oauthlib import OAuth1Session
flickr = OAuth1Session('462f8e29852bb3330a49734ae6cf1d59',
                            client_secret='ecc275b45d62e864',
                            resource_owner_key='72157720895409885-aa3e89a583876e51',
                            resource_owner_secret='e5316384df71ec1a')
url = 'https://www.flickr.com/services/rest?nojsoncallback=1&format=json&method=flickr.test.login'
r = flickr.get(url)

r.content

# COMMAND ----------

url = "https://www.flickr.com/services/rest/?method=flickr.photos.setMeta" 

photo_id = "53191900248"

extract = "dark with a black background and a white horse"

description = """dark with a black background and a white horseman, who is wearing his own colored belt. Both sides look like they\'ve never used anything before from other worlds. Like it was the first time ever that he rode back to North Carolina at these points in history when I said about him. In addition not appearing as awkward walking around my home or anywhere else until being taken away for several more weeks to be worn by this man while doing some basic chores on an old wooden platform out back of the building...the guys over there are staring me dead straight because what have you been up-side so far looking forward? Well guess once we get ridOf everything else then it\'s very likely their turn, that night our lives would change without any forefingering!  It has something different than just standing here. We walk into the store talking silently during one conversation whilst watching TV programmes which shows us where others go if we leave town tomorrow morning after bed tonight next door but then only saying, "I can buy coffee now" or whatever happens when someone sits right opposite me in the shop, or I have got another cup off myself with a bag full o\' ice cream from frozen eggs." Suddenly she smiles again. She says having breakfast this afternoon next Saturday but says sitting down is actually important since making decisions later goes longer daywise anyway - what do those men call decision Making....  You know...you think to wait longer till today...after two really long nights before deciding your plan has come complete— I don\'t understand all things; it is really simple though why I am thinking..."Okay no idea how much we had shared, let alone it does seem a lot easier than hoping everyone were happy after eating." I open top drawer knowing her secret gift will help pay a small fortune here or elsewhere...then I donot care whether or Not because in fact even with having dinner I could prefer going home". She laughs at this suggestion to me. He walks outside towards Tr"""

data={"photo_id":photo_id, "description":description}

r = flickr.post(url, data=data)

r.content

# COMMAND ----------

# Importing the required libraries
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# COMMAND ----------

extract = "there is a man riding a bike down a winding road"

# COMMAND ----------

from nltk import word_tokenize, pos_tag
nouns = [token for token, pos in pos_tag(word_tokenize(extract)) if pos.startswith('N')]
nouns

# COMMAND ----------

url = "https://www.flickr.com/services/rest/?method=flickr.photos.addTags" 

photo_id = "53191900248"
tags = 'dark, background, horse'
data={"photo_id":photo_id, "tags":tags}

r = flickr.post(url, data=data)

r.content

# COMMAND ----------


