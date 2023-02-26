FROM python:3.7

ADD . /MyfaceApp
WORKDIR /MyfaceApp

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
ENV LANG C.UTF-8
CMD ["python", "MyfaceApp.py"]