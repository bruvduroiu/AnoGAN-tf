#!/bin/bash
AWS_ACCESS_KEY_ID=$(aws --profile default configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws --profile default configure get aws_secret_access_key)

docker build -t anogan-tf .
~/.docker_aws/nvidia-docker run --rm \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    anogan-tf \
    --no-display \
    --g-epochs 30000 \
    --g-print-interval 1000 \
    --a-epochs 30000 \
    --a-print-interval 1000 \
    --outlier \
    --s3-bucket dissertation-backups \
    --s3-path results/anogan

unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
