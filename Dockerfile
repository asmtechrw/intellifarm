# The python image, used to build the virtual environment
FROM python:3.10.14-slim

LABEL maintainer="hrmuwanika@gmail.com"

RUN apt-get update && apt full-upgrade -y && apt-get autoremove -y

RUN apt-get install -y git wget python3-pip python3-opencv

RUN /usr/local/bin/python -m pip install --upgrade pip

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
