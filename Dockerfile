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

RUN pip install -r $APP_HOME/src/requirements.txt

COPY ./entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r//' /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN chown app_user /entrypoint.sh

WORKDIR $APP_HOME/src

ENTRYPOINT ["/entrypoint.sh"]
