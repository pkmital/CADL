#!/bin/bash
rm -rf session-1/tests/__pycache__
rm -rf session-2/tests/__pycache__
rm -rf session-3/tests/__pycache__
rm -rf session-4/tests/__pycache__
rm -rf session-5/tests/__pycache__
docker run -it -v /$(pwd)/session-1:/notebooks cadl py.test 
docker run -it -v /$(pwd)/session-2:/notebooks cadl py.test 
docker run -it -v /$(pwd)/session-3:/notebooks cadl py.test 
docker run -it -v /$(pwd)/session-4:/notebooks cadl py.test 
docker run -it -v /$(pwd)/session-5:/notebooks cadl py.test 
