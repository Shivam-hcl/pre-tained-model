FROM python:3.8-slim-buster
LABEL "usecase"="AI0002"
LABEL "usecase_test"="AI0002_test"

COPY  utility.py utility.py
COPY  __init__.py __init__.py
COPY  dataProfiler.py dataProfiler.py
COPY  data_profiler_functions.py data_profiler_functions.py
COPY  text text
COPY  utils utils
COPY  aionCode.py aionCode.py
COPY  requirements.txt requirements.txt
COPY  config.json config.json
RUN  pip install --no-cache-dir -r requirements.txt
