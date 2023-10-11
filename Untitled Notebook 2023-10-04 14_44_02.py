# Databricks notebook source
from flickr.flickr import *
from flickr.photos import *
from model.caption_to_story import *
from model.image_to_caption import *

# COMMAND ----------

f_conn = Flickr_conn()
f_conn.create_connection()
print(f_conn.conn)


# COMMAND ----------

USER = "131401388@N03"
IMAGE_NAME = "2023_Teracea_22"

photo_1 = Photos(image_name = IMAGE_NAME, user_id = USER)

# COMMAND ----------

photo_1.get_url(f_conn.conn)

print("url : ", photo_1.url)
photo_1.download_image()

# COMMAND ----------



# COMMAND ----------


