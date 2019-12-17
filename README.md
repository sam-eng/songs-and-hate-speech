# NLP Final Project

To set up and run with a virtual environment:

Generate an API key for Genius' API. Replace "[INSERT API KEY HERE]" with your API key.

> virtualenv venv
> source venv/bin/activate
> pip install -r requirements.txt

Data file guide:

- filtered-trump-tweets-with-lyrics.csv: list of tweets from trump_tweets identified as having song lyrics with offensive language
- filtered-tweets-with-lyrics.csv: list of tweets from labeled_data.csv identified as having song lyrics with offensive language
- labeled_data.csv: file of labeled tweets from Davidson et al's study
- notes.txt: titles of songs whose lyrics could either not be returned, were in the wrong language, or were not lyrics. Created manually.
- song-info-final.txt: the created data set containing songs, their artists, their lyrics, and n-grams
- trump_tweets.csv: file of test tweets
- trump-tweets-with-lyrics.csv: list of tweets from trump_tweets.csv identified as having song lyrics
- tweets-with-lyrics.csv: list of tweets from labeled_data.csv identified as having song lyrics

How to run:
python3 genius.py [data file name of tweets to match] [output file name to write results to]

Notes:
This code assumes that there is already a dataset called song-info-final.txt that contains JSON data described in our project write-up.