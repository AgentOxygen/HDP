FROM python:3.12-slim

WORKDIR /project

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pytest sphinx sphinx-autobuild

RUN pip install -e .


CMD ["pytest", "-v", "hdp/tests/"]