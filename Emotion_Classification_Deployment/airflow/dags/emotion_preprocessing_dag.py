import sys
import os
 
# Ensure the current directory is in sys.path so imports work
sys.path.append(os.path.dirname(__file__))
 
from datetime import datetime
 
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
# 🔁 Import your preprocessing function
from text_preprocessing import preprocess_csv
 
# === Fill these in from Azure ML Studio UI ===
DATASTORE_NAME = "workspaceblobstore"
BLOB_PATH = "test.csv"
LOCAL_FILE_NAME = "test.csv"
 
# === Local paths
LOCAL_PATH = f"/tmp/{LOCAL_FILE_NAME}"
PROCESSED_PATH = f"/tmp/processed_{LOCAL_FILE_NAME}"
 
 
def download_blob():
    conn = BaseHook.get_connection("azure_ml_conn")
    extras = conn.extra_dejson
 
    credential = ClientSecretCredential(
        tenant_id=extras["tenant_id"],
        client_id=extras["client_id"],
        client_secret=extras["client_secret"],
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=extras["subscription_id"],
        resource_group_name=extras["resource_group"],
        workspace_name=extras["workspace_name"],
    )
 
    datastore = ml_client.datastores.get(DATASTORE_NAME)
    account_name = datastore.account_name
    container_name = datastore.container_name
 
    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential,
    )
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(BLOB_PATH)
 
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    with open(LOCAL_PATH, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())
 
    print(f"Downloaded blob '{BLOB_PATH}' to '{LOCAL_PATH}'")
 
 
def preprocess_blob():
    print(f"Starting preprocessing on {LOCAL_PATH}")
    try:
        preprocess_csv(
            input_path=LOCAL_PATH,
            output_path=PROCESSED_PATH,
            text_column="text",
            keep_placeholders=False,
        )
        print(f"Preprocessed file saved to {PROCESSED_PATH}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise
 
 
def create_data_asset():
    conn = BaseHook.get_connection("azure_ml_conn")
    extras = conn.extra_dejson
 
    credential = ClientSecretCredential(
        tenant_id=extras["tenant_id"],
        client_id=extras["client_id"],
        client_secret=extras["client_secret"],
    )
 
    ml_client = MLClient(
        credential=credential,
        subscription_id=extras["subscription_id"],
        resource_group_name=extras["resource_group"],
        workspace_name=extras["workspace_name"],
    )
 
    data_asset = Data(
        path=PROCESSED_PATH,  # upload the processed file
        type="uri_file",
        description="Processed CSV registered via Airflow",
        name="processed_text_data"
    )
 
    ml_client.data.create_or_update(data_asset)
    print("Processed data asset created successfully.")
 
 
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}
 
with DAG(
    dag_id="azureml_text_preprocessing",
    default_args=default_args,
    schedule_interval="@weekly",  # Run once for demonstration
    catchup=False,
    tags=["azureml", "blob", "preprocessing"],
) as dag:
    download_task = PythonOperator(
        task_id="download_blob",
        python_callable=download_blob,
    )
 
    preprocess_task = PythonOperator(
        task_id="preprocess_blob",
        python_callable=preprocess_blob,
    )
 
    create_asset_task = PythonOperator(
        task_id="create_data_asset",
        python_callable=create_data_asset,
    )
 
    download_task >> preprocess_task >> create_asset_task