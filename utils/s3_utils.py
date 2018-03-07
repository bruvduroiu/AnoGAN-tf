import io
import os

import boto3

def savefig(figure, file_path='figure.png', bucket_name=None):
    if not bucket_name:
        raise Exception('You must provide an S3 bucket to store the figure.')

    img_data = io.BytesIO()
    figure.savefig(img_data, format='png')
    img_data.seek(0)

    s3 = boto3.client(
        's3',
        aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    )
    s3.put_object(
        Body=img_data,
        Bucket=bucket_name,
        ContentType='image/png',
        Key=file_path
    )

