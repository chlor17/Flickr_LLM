# Databricks notebook source
from flickr.flickr import *
from flickr.photos import *
from model.caption_to_story import *
from model.image_to_caption import *
import pandas as pd

f_conn = Flickr_conn()
f_conn.create_connection()

USER = "131401388@N03"
IMAGE_NAME = "2023_Montreal_243"

photo_1 = Photos(image_name = IMAGE_NAME, user_id = USER)

photo_1.get_url(f_conn.conn)

print("url : ", photo_1.url)
photo_1.download_image()

photo_1.print_image()

# COMMAND ----------

photo_1.photo_id

# COMMAND ----------

model_1 = image_to_caption()

photo_1.caption =  model_1.captioner(photo_1.get_image())

photo_1.caption

# COMMAND ----------

sess = Flickr_sess()
xml = sess.get_exif(photo_1)

photo_1.parse_xml(xml)

# COMMAND ----------

model_2 = caption_to_story()

model_2.set_api()

photo_1.story = model_2.create_story(photo_1)

photo_1.story

# COMMAND ----------

r = sess.get_photo_list(USER)
r

# COMMAND ----------

photo_list = xmltodict.parse(r)['rsp']['photos']['photo']
for i in photo_list[0:5]:
  print(i["@id"])

# COMMAND ----------

col_list = ['image_name',
      'photo_id',
      'user_id',
      'destination_path',
      'image_path',
      'url',
      'caption',
      'story',
      'aperture',
      'exposuretime',
      'iso',
      'createdate'] # ,
      # 'geo']

df = pd.DataFrame(columns=col_list)

row = {}
for col in col_list:
      row[col]  = getattr(photo_1, col)
df = pd.concat([df, pd.DataFrame.from_dict(row)], ignore_index=True)
spark_df_new = spark.createDataFrame(df)


# COMMAND ----------

# spark_df_new = spark.createDataFrame(df)
# spark_df.write.saveAsTable(
#   name = "chlor_catalog.flickr_llm.photo_table"
# )

# COMMAND ----------

spark_df = spark.read.table("chlor_catalog.flickr_llm.photo_table")
if len(spark_df.filter(spark_df.photo_id.contains(photo_1.photo_id)).collect()) == 0:
  spark_df_new.write.mode("append").insertInto("chlor_catalog.flickr_llm.photo_table")

# COMMAND ----------

sess = Flickr_sess()
# r = sess.post_description(photo_1)

# COMMAND ----------

#install dependencies
import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import keras.utils as image_utils
# from imagenet_utils import decode_predictions
from keras.applications import imagenet_utils
from IPython.display import display, Image

# COMMAND ----------

#import the InceptionV3 model from the Keras applications module
print("[INFO] loading network...")
model = InceptionV3(include_top=True, weights='imagenet')
print("Model loaded.")


#Now define a function that resizes the input image to 299x299 pxiels and then converts
#the image to a NumPy array
def processing(im):
    img = image_utils.load_img(im, target_size=(299, 299))
    img = image_utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# COMMAND ----------

photo_1.destination_path

# COMMAND ----------

pred = model.predict(processing("/Volumes/chlor_catalog/flickr_llm/photos/2023_Teracea_236"))
print('Predicted:', imagenet_utils.decode_predictions(pred))

# COMMAND ----------

pred

# COMMAND ----------

/Volumes/chlor_catalog/flickr_llm/photos/2023_Montreal_188.jpg
