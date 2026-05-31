FROM apache/airflow:2.8.3

USER airflow

COPY requirements.txt /tmp/requirements.txt

COPY .env /opt/airflow

RUN pip install --no-cache-dir --user -r /tmp/requirements.txt
