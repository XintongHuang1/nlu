# run_finetune.py

import time
import argparse
from openai import OpenAI

API_KEY = "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"  
client = OpenAI(api_key=API_KEY)

# Train and Validation/test path
train_file_path = "anchoring_train.jsonl"
val_file_path = "anchoring_test.jsonl"

def upload_file(file_path, purpose="fine-tune"):
  with open(file_path, "rb") as f:
    response = client.files.create(file=f, purpose=purpose)
    print(f"Upload successfully: {file_path} -> {response.id}")
    return response.id

def create_fine_tune_job(train_id, val_id, base_model="gpt-3.5-turbo"):
    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=base_model
    )
    print(f"Fine-tuning job started! Job ID: {job.id}")
    return job.id

def wait_until_complete(job_id):
    print("Waiting for fine-tuning to complete...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"Status: {status}")
        if status == "succeeded":
            print(f"Fine-tuning complete! Model ID: {job.fine_tuned_model}")
            break
        elif status == "failed":
            print("Fine-tuning failed. Please check your dataset or try again.")
            break
        time.sleep(20)

# Add this for CLI argument support
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Base model to fine-tune")
    args = parser.parse_args()

    train_id = upload_file(train_file_path)
    val_id = upload_file(val_file_path)
    job_id = create_fine_tune_job(train_id, val_id, base_model=args.model)
    wait_until_complete(job_id)
