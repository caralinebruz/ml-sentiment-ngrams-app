FROM python:3

ENV PORT 8000
EXPOSE $PORT

WORKDIR /app

ENV MODEL_DIR /model

ADD requirements.txt /app
RUN pip3 install -r requirements.txt


# do this last since ill be changing this most frequently as i develop
ADD web-app/. /app


# Run the service
CMD [ "python", "web.py" ]


# docker build -t web-app .
# docker run -p 8000:8000 -it web-app
