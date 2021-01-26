#!/bin/bash

base=data/COVID-19-TweetIDs
root=$(dirname $base)
#FILES=${base}/2020-0[5,6]/*
FILES=${base}/2020-10/*
for f in $FILES
do
	echo $f
	echo $root/sampled_ids/${f#$base/}
	cat $f | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .01) print $0}' > $root/sampled_ids/${f#$base/}
done