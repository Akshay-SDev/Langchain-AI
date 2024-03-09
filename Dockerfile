FROM python:3.11-slim

ENV HOME /root
ENV APP_HOME /application

RUN ["apt-get", "clean"]
RUN ["apt-get", "update"]

RUN groupadd -r app_user && useradd -r -g app_user app_user

RUN mkdir -p $APP_HOME

COPY . $APP_HOME
RUN chown -R app_user $APP_HOME

RUN mkdir -p /logs
RUN chown -R app_user:app_user /logs

RUN pip install -r $APP_HOME/requirements.txt
