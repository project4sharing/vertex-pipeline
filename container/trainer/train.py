from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from google.cloud import bigquery
from google.cloud import storage
from joblib import dump

import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"#####  Environment variables:")
for key, value in sorted(os.environ.items()):
    logger.info(f"#####  {key}={value}")

bqclient = bigquery.Client()
storage_client = storage.Client()

def download_table(bq_table_uri: str):
    prefix = "bq://"
    if bq_table_uri.startswith(prefix):
        bq_table_uri = bq_table_uri[len(prefix):]

    table = bigquery.TableReference.from_string(bq_table_uri)
    rows = bqclient.list_rows(
        table,
    )
    return rows.to_dataframe(create_bqstorage_client=False)

# These environment variables are from Vertex AI managed datasets
training_data_uri = os.environ["AIP_TRAINING_DATA_URI"]
test_data_uri = os.environ["AIP_TEST_DATA_URI"]

logger.info(f"Training data URI: {training_data_uri}")
logger.info(f"Test data URI: {test_data_uri}")

# Download data into Pandas DataFrames, split into train / test
df = download_table(training_data_uri)
test_df = download_table(test_data_uri)
labels = df.pop("Class").tolist()
data = df.values.tolist()
test_labels = test_df.pop("Class").tolist()
test_data = test_df.values.tolist()

# Define and train the Scikit model
skmodel = DecisionTreeClassifier()
skmodel.fit(data, labels)
score = skmodel.score(test_data, test_labels)
logger.info(f'#####  accuracy is: {score}')

# Save the model to a local file
try:
    dump(skmodel, "model.joblib")
except Exception as e:
    logger.error(f"#####  Error occurred while saving the model: {e}")

# Upload the saved model file to GCS
try:
    logger.info(f"#####  AIP_STORAGE_BUCKET:: {os.environ['AIP_STORAGE_BUCKET']}")
    bucket = storage_client.get_bucket(os.environ["AIP_STORAGE_BUCKET"])
    logger.info(f"#####  AIP_MODEL_DIR:: {os.environ['AIP_MODEL_DIR']}")
    model_directory = os.environ["AIP_MODEL_DIR"]
    storage_path = os.path.join(model_directory, "model.joblib")
    logger.info(f"#####  storage_path:: {storage_path}")
    blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
    blob.upload_from_filename("model.joblib")
except Exception as e:
    logger.error(f"#####  Error occurred while uploading model to GCS: {e}")