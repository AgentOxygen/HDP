FROM python:latest

WORKDIR /project

RUN adduser --disabled-password hdp

COPY . .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

RUN chown -R hdp:hdp /project

USER hdp

CMD ["pytest", "-v", "hdp/tests/"]