# syntax=docker/dockerfile:1 
FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev build-essential libssl-dev libffi-dev


# install app
COPY . . 
RUN pip install --no-cache-dir -r requisitos.txt
# final configuration
EXPOSE 8050
CMD  ["python3", "tablero.py", "--host", "0.0.0.0", "--port", "8050"]