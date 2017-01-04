#!/bin/bash
docker run -it -v /$(pwd)/session-1:/notebooks cadl py.test --cache-clear 
docker run -it -v /$(pwd)/session-2:/notebooks cadl py.test --cache-clear 
docker run -it -v /$(pwd)/session-3:/notebooks cadl py.test --cache-clear 
docker run -it -v /$(pwd)/session-4:/notebooks cadl py.test --cache-clear 
docker run -it -v /$(pwd)/session-5:/notebooks cadl py.test --cache-clear 
