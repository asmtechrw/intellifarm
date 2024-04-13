FROM python:3.8.12-slim

RUN /usr/local/bin/python -m pip install --upgrade pip

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8503

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
