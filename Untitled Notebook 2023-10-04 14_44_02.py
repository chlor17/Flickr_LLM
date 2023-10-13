# Databricks notebook source
from flickr.flickr import *
from flickr.photos import *
from model.caption_to_story import *
from model.image_to_caption import *

# COMMAND ----------

f_conn = Flickr_conn()
f_conn.create_connection()

USER = "131401388@N03"
IMAGE_NAME = "Sep 13 2023_001"

photo_1 = Photos(image_name = IMAGE_NAME, user_id = USER)

# COMMAND ----------

photo_1.get_url(f_conn.conn)

print("url : ", photo_1.url)
photo_1.download_image()

# COMMAND ----------

photo_1.print_image()

# COMMAND ----------

model_1 = image_to_caption()

# COMMAND ----------

photo_1.caption =  model_1.captioner(photo_1.get_image())

# COMMAND ----------

photo_1.caption[0]['generated_text']

# COMMAND ----------

model_2 = caption_to_story()

model_2.set_api()

photo_1.story = model_2.create_story(photo_1.caption[0]['generated_text'])



# COMMAND ----------

photo_1.story

# COMMAND ----------

sess = Flickr_sess()
r = sess.post_description(photo_1)

# COMMAND ----------

r.content

# COMMAND ----------


