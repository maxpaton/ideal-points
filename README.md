# Ideal Points: COVID-19 Twitter Analysis

In this project, I quantify various prototypical authors' (i.e. politicians, doctors, etc.) stance on COVID-19, by analysing polarisation in tweets. The tweets used were fetched using a random sample of COVID-related tweet IDs from [https://github.com/echen102/COVID-19-TweetIDs](https://github.com/echen102/COVID-19-TweetIDs), spanning over six months.

An 'ideal points' model, which is an unsupervised probabalistic topic model, is used to detect topics as well as quantifying the authors' stance on an interpretable scale. This model, which is in the [tbip](https://github.com/maxpaton/ideal-points/tree/main/tbip) directory, was taken from [https://github.com/keyonvafa/tbip](https://github.com/keyonvafa/tbip), and has only been adapted slightly for this project.

Since using the 'ideal points' model in the traditional way (using textual information from various specific authors) would render the problem impractical for this task (there are a vast number of authors/Twitter users). I had to generalise the model. In order to adapt the model for my purpose, prototypical author profiles were built up from using extensive NLP techniques to analyse the bios corresponding to the tweets, which I used to created four broad profiles: academics, doctors, journalists and politicians. 
