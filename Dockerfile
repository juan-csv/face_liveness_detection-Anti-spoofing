FROM python:3.7-slim-buster
WORKDIR /home
COPY . /home
RUN apt-get update && apt-get install -y libglib2.0-0 \
	build-essential \
	cmake \
	pkg-config \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libsm6 \
	libxext6 \ 
	libxrender-dev
RUN pip install -r requeriments.txt
CMD python face_anti_spoofing.py
