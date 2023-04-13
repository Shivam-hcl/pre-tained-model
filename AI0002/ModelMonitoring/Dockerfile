FROM python:3.8-slim-buster
LABEL "usecase"="AI0002"
LABEL "usecase_test"="AI0002_test"

COPY  utility.py utility.py
COPY  data_reader.py data_reader.py
COPY  __init__.py __init__.py
COPY  aionCode.py aionCode.py
COPY  input_drift.py input_drift.py
COPY  config.json config.json
COPY  requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt