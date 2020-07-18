import configparser
from datetime import datetime
import os
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql import types as t
from pyspark.sql.types     import IntegerType
from pyspark.sql.functions import to_timestamp
import boto3
import numpy as np
import pandas as pd
import time
import datetime


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']

"""^^^Get AWS Access Key ID & Secret Key^^^"""


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark
print('Creating Spark session: COMPLETE')

"""^^^Creating Spark session for data processing, if it does not currently exist^^^"""


def process_song_data(spark, input_data, output_data):
    print('Processing song_data from S3 bucket...')
    """
    This function will:  Extract and process song data to 
    create 2 dimensional tables (songs_table & artists_table)
    """
    song_data = f'{input_data}/song_data/A/A/A/*.json'
    df = spark.read.json(song_data)
    print('Reading song_data from S3 bucket: COMPLETE')
    
    """^^^Read in song_data from Sparkify S3 bucket and assign it as df^^^"""
    
    songs_table = df.select('song_id', 'title', 'artist_id', 
                            'year', 'duration').dropDuplicates()
    songs_table.printSchema()
    songs_table.show(5)
    songs_table.write.parquet(f'{output_data}/songs_table',
                              mode='overwrite',
                             partitionBy=['year','artist_id'])
    print('Write songs_table to parquet files & partition by year & artist_id: COMPLETE')
    
    """^^^Extract columns('song_id', 'title', 'artist_id', 'year', 'duration') 
    from df and assign it to songs_table.
    Write songs_table to parquet files for output and partition by 'year','artist_id'^^^"""
    
    
    artists_table = df.select('artist_id', 'artist_name',
                              'artist_location', 'artist_latitude',
                              'artist_longitude').dropDuplicates()
    artists_table.printSchema()
    artists_table.show(5)
    artists_table.write.parquet(f'{output_data}/ artists_table',
                                   mode='overwrite')
    print('Write artists_table to parquet files: COMPLETE')
    
    """^^^Extract columns('artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude') from df and assign it to artists_table.
    Write artists_table to parquet files for output^^^"""
    


def process_log_data(spark, input_data, output_data):
    print('Processing log_data from S3 bucket...')
    
    """This function will:  Extract and process log data"""
    
    log_data = f'{input_data}/log_data/*/*/*.json'
    log_df = spark.read.json(log_data)
    print('Reading log_data from S3 bucket: COMPLETE')
    
    """^^^Read in log_data from Sparkify S3 bucket and assign it as log_df^^^"""

    
    log_df = log_df.filter(log_df['page'] == 'NextSong')
    users_table = log_df.select('userId', 'firstName', 'lastName',
                           'gender', 'level').dropDuplicates()
    users_table.printSchema
    users_table.show(5)
    users_table.write.parquet(f'{output_data}/users_table', mode='overwrite')
    print('Write users_table to parquet files: COMPLETE')
    
    """^^^Extract columns('userId', 'firstName', 'lastName', 'gender', 'level') 
    from log_df and assign it to users_table.
    Write users_table to parquet files for output^^^"""
    
    
    get_timestamp = udf(lambda x: datetime.datetime.fromtimestamp(int(x / 1000)) \
                                          .strftime('%Y-%m-%d %H:%M:%S'))
    log_df = log_df.withColumn( "timestamp"
                                         , to_timestamp(get_timestamp(log_df.ts)))
    get_datetime = udf(lambda x: datetime.datetime.fromtimestamp(int(x / 1000)) \
                                         .strftime('%Y-%m-%d %H:%M:%S'))
    log_df = log_df.withColumn( "datetime"
                                         , get_datetime(log_df.ts))
    time_table = log_df.select \
                    ( col('timestamp').alias('start_time')
                    , hour('datetime').alias('hour')
                    , dayofmonth('datetime').alias('day')
                    , weekofyear('datetime').alias('week')
                    , month('datetime').alias('month')
                    , year('datetime').alias('year')
                    , date_format('datetime', 'F').alias('weekday'))
    print('Creating datetime column from timestamp: COMPLETE')
    time_table.write.parquet(f'{output_data}/time_table',
                             mode='overwrite',
                             partitionBy=['year', 'month'])
    print('Writing time_table to parquet: COMPLETE')
    
    """^^^Creating datetime column from timestamp.
    Extract columns('hour', 'day', 'week', 'month', 'year', 'weekday') 
    from log_df and assign it to time_table.
    Write time_table to parquet files for output^^^"""
    
    
    song_df = spark.read.json(input_data + "song_data/A/A/A/*.json")
    songplays_table = log_df \
        .join( song_df
                , (log_df.song   == song_df.title) & \
                (log_df.artist == song_df.artist_name)
                , 'left_outer') \
        .select( col("timestamp").alias("start_time")
                , col("userId").alias("user_id")
                , log_df.level
                , song_df.song_id
                , song_df.artist_id
                , col("sessionId").alias("session_id")
                , log_df.location
                , col("useragent").alias("user_agent")
                , year("datetime").alias("year")
                , month("datetime").alias("month") )
    songplays_table.write.parquet(f'{output_data}/songplays_table', mode='overwrite',
                                     partitionBy=['year', 'month'])
    print('Write songplays_table to parquet files & partition by year and month: COMPLETE')
    
    """^^^Read in song_data to create songplays_table.
    Extract columns and assign it to songplays_table.
    Write songplays_table to parquet files for output^^^"""
    
    
    
def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://p4-buckey/" 
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)
    
    print('ETL PROCESSING COMPLETE!')

if __name__ == "__main__":
    main()
