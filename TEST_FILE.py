# Databricks notebook source
from flickr.flickr import *
from flickr.photos import *
from model.caption_to_story import *
from model.image_to_caption import *
from utils import read_yaml

# COMMAND ----------

config_file = read_yaml()


# COMMAND ----------

f_conn = Flickr_conn()
f_conn.create_connection()
print(f_conn.conn)
photo_1 = Photos(image_name = IMAGE_NAME, user_id = USER)

# COMMAND ----------


