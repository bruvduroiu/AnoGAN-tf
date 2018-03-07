FROM tensorflow/tensorflow:latest-gpu-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /usr/src/app /usr/src/app/anogan /usr/src/app/weights /usr/src/app/utils
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app
RUN pip install -r requirements.txt

COPY main.py /usr/src/app
COPY anogan/ /usr/src/app/anogan/
COPY utils/ /usr/src/app/utils/

ENTRYPOINT [ "python", "main.py" ]