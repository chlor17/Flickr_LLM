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
        self.caption = None
        self.story = None


    def get_url(self, conn): # should be only 1, assumes all image name are different
        photos_list = conn.walk(text=self.image_name,
                              user_id=self.user_id,
                              extras= ','.join(self.photo_size),
                              privacy_filter=1, 
                              per_page=1,
                              sort='relevance')
        try:
            for photo in photos_list:
                for size in range(len(self.photo_size)):  # makes sure the loop is done in the order we want
                    self.url = photo.get(self.photo_size[size])
        except:
            print("Photo list seems empty")

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
      display(image)

    def get_image(self):
      return Image.open(os.path.join(self.destination_path, self.image_name)) .convert("RGB")
  
    def get_image_id(self):
        return self.url.split('/')[4].split('_')[0]

