# Databricks notebook source
from flickr.flickr import *
from flickr.photos import *
from model.caption_to_story import *
from model.image_to_caption import *
import pandas as pd

USER = "131401388@N03"

# COMMAND ----------

sess = Flickr_sess()
xml_photo_list = sess.get_photo_list(USER)
photo_list = xmltodict.parse(xml_photo_list)['rsp']['photos']['photo']
spark_df = spark.read.table("chlor_catalog.flickr_llm.photo_table")

new_image = True
i = 0
while new_image: # len(photo_list):
    photo_detail = photo_list[i]
    f_conn = Flickr_conn()
    f_conn.create_connection()
    IMAGE_NAME = photo_detail['@title']
    photo = Photos(image_name = IMAGE_NAME, user_id = USER)
    photo.get_url(f_conn.conn)
    photo.download_image()
    model_caption = image_to_caption()
    photo.caption =  model_caption.captioner(photo.get_image())

    xml = sess.get_exif(photo)
    photo.parse_xml(xml)
    model_desc = caption_to_story()
    model_desc.set_api()
    photo.story = model_desc.create_story(photo)

    if len(spark_df.filter(spark_df.photo_id.contains(photo.photo_id)).collect()) == 0:
        spark_df_new = create_df_from_photo(photo)
        spark_df_new.write.mode("append").insertInto("chlor_catalog.flickr_llm.photo_table")
        sess.post_description(photo)
    else:
        new_image = False
    i+=1
