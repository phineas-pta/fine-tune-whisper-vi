#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test Amazon Transcribe

only test some small files, because test everything will cost several thousands of dollars.

copied from https://github.com/Picovoice/speech-to-text-benchmark/blob/master/engine.py >>> AmazonTranscribeEngine
"""

import os.path, uuid, json, requests, time, boto3

# AWS account > security credentials > access keys
ACCESS_KEY = "███"
SECRET_KEY = "███"
AWS_REGION = "us-west-2"

S3_CLIENT = boto3.client("s3", region_name=AWS_REGION, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
TRANS_CLIENT = boto3.client("transcribe", region_name=AWS_REGION, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

S3_BUCKET = str(uuid.uuid4())
S3_CLIENT.create_bucket(ACL="private", Bucket=S3_BUCKET, CreateBucketConfiguration={"LocationConstraint": AWS_REGION})


def transcribe(audio_file: str) -> str:
	s3_object = os.path.basename(audio_file)
	S3_CLIENT.upload_file(audio_file, S3_BUCKET, s3_object)

	trans_job_name = str(uuid.uuid4())
	trans_job = TRANS_CLIENT.start_transcription_job(
		TranscriptionJobName=trans_job_name, LanguageCode="vi-VN", MediaFormat="wav",
		Media={"MediaFileUri": f"https://s3-{AWS_REGION}.amazonaws.com/{S3_BUCKET}/{s3_object}"}
	)["TranscriptionJob"]

	while trans_job["TranscriptionJobStatus"] != "COMPLETED":
		trans_job = TRANS_CLIENT.get_transcription_job(TranscriptionJobName=trans_job_name)["TranscriptionJob"]
		time.sleep(3)

	content = requests.get(trans_job["Transcript"]["TranscriptFileUri"])
	return json.loads(content.content.decode("utf8"))["results"]["transcripts"][0]["transcript"]


# select random audio from FLEURS
transcribe("audio_fleurs_1709.wav")
transcribe("audio_fleurs_1689.wav")
transcribe("audio_fleurs_1738.wav")


# [x["Name"] for x in S3_CLIENT.list_buckets()["Buckets"]]
response = S3_CLIENT.list_objects_v2(Bucket=S3_BUCKET)
while response["KeyCount"] > 0:
	S3_CLIENT.delete_objects(
		Bucket=S3_BUCKET,
		Delete={"Objects": [{"Key": obj["Key"]} for obj in response["Contents"]]}
	)
	response = S3_CLIENT.list_objects_v2(Bucket=S3_BUCKET)
S3_CLIENT.delete_bucket(Bucket=S3_BUCKET)
