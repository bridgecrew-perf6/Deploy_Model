FROM python:3

WORKDIR C:\Users\Admin\PycharmProjects\flask_tutorial

COPY . .

RUN pip install -r requirements.txt

CMD ["python","./tensorflowjs_serve.py"]