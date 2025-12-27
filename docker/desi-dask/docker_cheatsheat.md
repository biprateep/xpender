### Build image on my laptop (once in the docker directory)
docker build -t biprateep/desi-dask:latest .

### Push image to Docker Hub
docker push biprateep/desi-dask:latest 

### Pull image into NERSC
shifterimg -v pull biprateep/desi-dask:latest

### Run interactive shell in the image
shifter --image=biprateep/desi-dask:latest /bin/bash

### NERSC Documentation for jupyter with docker/shifter
https://docs.nersc.gov/services/jupyter/how-to-guides/#shifter