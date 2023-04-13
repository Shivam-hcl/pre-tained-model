FROM python:3.8-slim-buster
LABEL "usecase"="AI0002"
LABEL "usecase_test"="AI0002_test"

COPY  utility.py utility.py
COPY  data_reader.py data_reader.py
COPY  text text
COPY  utils utils
COPY  __init__.py __init__.py
COPY  predict.py predict.py
COPY  groundtruth.py groundtruth.py
COPY  aionCode.py aionCode.py
COPY  requirements.txt requirements.txt
COPY  config.json config.json
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "aionCode.py","-ip","0.0.0.0","-pn","8094"]
