#!/bin/bash
find ../covid_dev/app/static/results -mindepth 1 -maxdepth 1 -type d -ctime +3 | xargs rm -rf
find ../covid_dev/app/static/img/output -mindepth 1 -maxdepth 1 -ctime +1 | xargs rm
find ../covid_dev/app/static/csv/uploads -mindepth 1 -maxdepth 1 -ctime +1 | xargs rm
find ../covid_dev/app/static/csv/configs -mindepth 1 -maxdepth 1 -ctime +1 | xargs rm
