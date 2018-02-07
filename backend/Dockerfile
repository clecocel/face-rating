FROM tiangolo/uwsgi-nginx-flask:python3.5

RUN apt-get -qq update
RUN apt-get install -y \
    man \
    --fix-missing

COPY ./backend/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

RUN pip install h5py
RUN pip install Pillow

# UWSGI and NGINX configs
COPY ./backend/config/uwsgi.ini /app/uwsgi.ini
COPY ./backend/config/nginx.conf /etc/nginx/conf.d/nginx.conf

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/usr/bin/supervisord"]

EXPOSE 8080


