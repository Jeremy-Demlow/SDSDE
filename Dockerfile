FROM python:3.9.1
ADD . /sdsde_library
RUN pip install -r /sdsde_library/requirements.txt
RUN pip install /sdsde_library/
