#!/bin/bash

FILES=COVID-19-TweetIDs/2020-0[5,6]/*
for f in $FILES
do
	echo "${f#*/}"
	echo $f
	cat $f | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .01) print $0}' > sampled_ids/${f#*/}
done
