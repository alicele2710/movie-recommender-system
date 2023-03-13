# -*- coding: utf-8 -*-
"""Data Pre-processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HxQunm74q3hPe-wlG63-IrMLvAk2LTI5
"""

from google.colab import auth
auth.authenticate_user()
from google.colab import drive
drive.mount('/content/drive')
!pip install pyspark

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import random
from pyspark.sql.functions import col
from pyspark.sql.types import StringType,BooleanType,DateType, StructType
from pyspark.sql.functions import split
from pyspark.sql.functions import *
from pyspark.sql.functions import window
from pyspark.sql.window import Window

spark = SparkSession\
.builder\
.appName('AIE')\
.master("local[*]")\
.config("spark.executor.memory", "70g")\
.config("spark.driver.memory", "50g")\
.config("spark.memory.offHeap.enabled",True)\
.config("spark.memory.offHeap.size","16g")\
.appName("sampleCodeForReference")\
.getOrCreate()

df = spark.read.format("csv").option("header","true").load("/content/drive/MyDrive/Class notes/machine learning in production/Group Project/sample_data.csv")

df.show()

df_viewing = df.where(~F.col('message').contains('/rate/'))

df_rating = df.where(F.col('message').contains('/rate/'))

#create movieid and watch_time columns
df_viewing = df_viewing.withColumn('movieid', split(df_viewing['message'], '/').getItem(3)) \
       .withColumn('watchtime', split(df_viewing['message'], '/').getItem(4))\
       .drop('message')
df_viewing = df_viewing.withColumn('watch_time', split(df_viewing['watchtime'],'\\.').getItem(0))\
       .drop('watchtime')
df_viewing.show(truncate=False)

#keep the highest watch_time per user_id and movieid
w = Window.partitionBy('user_id','movieid')
df_viewing = df_viewing.withColumn('max_watch_time', F.max('watch_time').over(w))\
    .where(F.col('watch_time') == F.col('max_watch_time'))\
    .drop('max_watch_time')\
    .drop('_c0')\
    .drop('time')
df_viewing.show()

#create rating column
df_rating = df_rating.withColumn('movieid_1', split(df_rating['message'], '/').getItem(2))
df_rating = df_rating.withColumn('movieid', split(df_rating['movieid_1'],'=').getItem(0))\
         .withColumn('rating', split(df_rating['movieid_1'], '=').getItem(1))\
         .drop('movieid_1')\
         .drop('message')\
         .drop('_c0')\
         .drop('time')
df_rating.show()

#check for duplicates
print((df_rating.count(), df_rating.distinct().count()))

print((df_viewing.count(), df_viewing.distinct().count()))

#join the 2 tables
df_result = df_viewing.join(df_rating,(df_viewing.user_id ==  df_rating.user_id)&(df_viewing.movieid == df_rating.movieid),'left') \
                      .select(df_viewing.user_id, df_viewing.movieid, df_viewing.watch_time, df_rating.rating)
df_result.show()

