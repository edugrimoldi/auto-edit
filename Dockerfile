FROM tensorflow/tensorflow:2.10.0

WORKDIR /auto-edit

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY autoedit autoedit
COPY src src
COPY setup.py setup.py
RUN pip install .

CMD uvicorn src.main:app --host 0.0.0.0 --port 8000