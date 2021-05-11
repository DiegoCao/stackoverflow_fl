import pandas as pd
import bq_helper
from bq_helper import BigQueryHelper


    

stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")

bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")

bq_assistant.list_tables()


