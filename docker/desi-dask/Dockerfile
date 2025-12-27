# Based on: https://github.com/legacysurvey/imagine/blob/493599da387eea664e132cb798500bf9bced7ed5/docker/cutouts/Dockerfile

FROM legacysurvey/legacypipe:DR10.3.6b

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt


#Install some of the DESI (latest version) packages which do not need path setups
RUN pip install --break-system-packages \
    git+https://github.com/desihub/desiutil.git#egg=desiutil \
    git+https://github.com/desihub/desimodel.git#egg=desimodel \
    git+https://github.com/desihub/desitarget.git#egg=desitarget \
    git+https://github.com/desihub/desispec.git#egg=desispec \
    git+https://github.com/desihub/speclite.git#egg=speclite

# Setup legacy survey cutout service and make it available
RUN git clone https://github.com/legacysurvey/imagine.git
RUN cd imagine/viewer && ln -s settings_test.py settings.py
RUN cd imagine && mv data data-orig && ln -s /dvs_ro/cfs/cdirs/cosmo/webapp/viewer/data .
COPY cutout /usr/local/bin/
RUN chmod 755 /usr/local/bin/cutout
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python:/app/imagine




#pinning dask to make current version of rechunker to work 
#(remove when new version of rechunker is released)
# RUN python3 -m pip install "dask[complete]==2024.11.2"

#This one does not use MPI and uses only TCP
# might break rechunker
RUN python -m pip install --break-system-packages "dask[complete]" 

#For future experimentation with dask-mpi
# RUN pip install dask_mpi --upgrade 



LABEL Maintainer="Biprateep Dey biprateep@pitt.edu"
