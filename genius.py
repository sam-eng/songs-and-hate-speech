'''
genius.py utilizes a billboard chart downloader (install with pip install billboard.py) to scrape the past 29 years of top rap/hip-hop
songs as ranked by Billboard, then uses Genius' API to build a dataset of lyrics from the songs.
'''

from requests import get
import billboard
import time
import sys
from requests.exceptions import HTTPError
from requests.exceptions import ConnectionError
import lyricsgenius
import nltk
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.collocations import *
from nltk.probability import FreqDist
import json
import pandas as pd
'''

Current issues: having trouble creating data set due to rate limits on server. Not quite sure what the limits are, so it's challenging.

Issue: timeout means I had to form song list/lyrics in chunks

TODO: Store lyrics (and n-grams) in text files

Return: list of tweets that contain a 4+ word phrase identified as characteristic of song lyrics

'''

# Using Billboard.py, get information of at least 150 songs in the Billboard hot rap tracks chart
def find_songs():
    r_songs = {}
    r_chart = billboard.ChartData('hot-rap-tracks')
    while r_chart.previousDate:
        for song in r_chart:
            r_songs[song.title] = song.artist
        try:
            r_chart = billboard.ChartData('hot-rap-tracks', r_chart.previousDate)
            time.sleep(2)
            print(r_chart.previousDate)
            # If date limit is reached, there are no more dates, or 150 song titles have been collected, stop.
            if (r_chart.previousDate is not None and r_chart.previousDate < "1990-01-01"):
                print("DONE")
                with open('song-titles.txt', 'a+') as f:
                    for song in r_songs:
                        f.write(song+"\t"+r_songs[song]+"\n")
                break
            if r_chart.previousDate is None:
                print("DONE: NO MORE DATES")
                with open('song-titles.txt', 'a+') as f:
                    for song in r_songs:
                        f.write(song+"\t"+r_songs[song]+"\n")
                break
            if (len(r_songs) >= 150):
                with open('song-titles.txt', 'a+') as f:
                    for song in r_songs:
                        f.write(song+"\t"+r_songs[song]+"\n")
                print("final (prev) date: " + r_chart.previousDate)
                break
        except HTTPError as err:
            print("Error reached, waiting 2 minutes to try again... ") 
            time.sleep(120)
            print("current song list size: " + str(len(r_songs)))
            '''with open('song-titles.txt', 'a+') as f:
                    for song in r_songs:
                        f.write(song+"\t"+r_songs[song]+"\n")
            print("final (prev) date: " + r_chart.previousDate)'''
            #break
        except (ConnectionResetError, ConnectionError) as e:
            with open('song-titles.txt', 'a+') as f:
                    for song in r_songs:
                        f.write(song+"\t"+r_songs[song]+"\n")
            print("CONNECTION ERROR")
            print("final (prev) date: " + r_chart.previousDate)
            break

# Using LyricsGenius, fetch stored information about songs, search for their lyrics in 
def find_lyrics():
    # Fetch saved song information from song-info.txt and save it as data
    with open("song-info.txt") as f:
         data = json.load(f)

    # Get songs from song-titles.txt, each row is a song title, followed by \t, then the artist
    with open("song-titles.txt") as f:
        s_data = f.read()
        songs = s_data.split("\n")
    for song in songs:
        info = song.split("\t")
        if info[0] == "Sunflower (Spider-Man: Into The Spider-Verse)":
            info[0] = "Sunflower"
        if (info[0] not in data):
            data[info[0]] = {}
            data[info[0]]["author"] = info[1]

    # bad_lyrics stores the titles of songs whose lyrics are wrong or missing
    bad_lyrics = []
    with open('notes.txt') as f:
        titles = f.read().split("\n")
        for title in titles:
            if (not title.startswith("IGNORE:")):
                bad_lyrics.append(title)
            else:
                continue
    # TODO: hide API key
    genius = lyricsgenius.Genius("xNYwq4ZDn6Ytqu7pJ7Jvy-84TIuWLgwQTDQZ7cMe1ykdDIY7OwSCnblRToxw5N1y")
    lyrics = {}
    
    for song in data:
        try: 
            if ("lyrics" not in data[song]):
                g_song = genius.search_song(song, data[song]["author"])
                if g_song is None:
                    print("ERROR: Lyrics could not be found")
                    with open("song-title-issues.txt", "a+") as f:
                        f.write(song + "\n")
                    continue
                lyrics[song] = g_song.lyrics
                print(song)
            elif song in bad_lyrics:
                g_song = genius.search_song(song, data[song]["author"])
                if g_song is None:
                    print("ERROR: Bad lyrics re-search failed")
                    with open("song-title-issues.txt", "a+") as f:
                        f.write(song + "\n")
                    continue
                lyrics[song] = g_song.lyrics
                print(g_song.lyrics)
                print(song)
            else:
                continue
        except: 
            print("ERROR: Could not get song lyrics")
            with open("song-title-issues.txt", "a+") as f:
                f.write(song + "\n")
            continue

    # Process lyrics
    for song in data:
        new_lyrics = []
        # Remove lines entirely in brackets and parenthesis
        for i in range(len(split_lyrics)):
            line = split_lyrics[i]
            if len(line) == 0:
                continue
            elif line[0] == "[" and line[len(line) - 1] == "]":
                continue
            elif line[0] == "(" and line[len(line) - 1] == ")":
                continue
            else:
                new_lyrics.append(line)
        #print(new_lyrics)
        # Save these lightly processed lyrics in the dictionary
        data[song]["lyrics"] = new_lyrics
        print(song)
    with open("song-info-2.txt", "w+") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    for song in data:
        if not "lyrics" in data[song]:
            continue
        if song in bad_lyrics:
            continue
        print(song)
        
        parsed_lyrics = []
        for i in range(len(data[song]["lyrics"])):
            line = data[song]["lyrics"][i]
            line = re.sub(r'\([^()]*\)', '', line)
            # REMOVE PUNCTUATION
            line = line.translate(str.maketrans('', '', string.punctuation.replace("-", ""))) 
            line = re.sub(r'([Yy]eah)|([Ww]hoah?)|([Oo]h)', '', line).strip() #won't remove ooh, remove Ha's
            line = line.lower()
            if len(line) != 0:
                parsed_lyrics.append(line)
        all_ngrams = []
        final_lyrics = " asdf ".join(parsed_lyrics)
        vect = CountVectorizer(ngram_range=(5, 8))
        analyzer = vect.build_analyzer()
        listNgramQuery = analyzer(final_lyrics)
        listNgramQuery.reverse()
        add = {}
        add[4] = []
        add[5] = []
        add[6] = []
        add[7] = []
        # Generate top 3 n-grams of each length for the song
        for ngram in listNgramQuery:
            ngrambits = ngram.split("asdf")
            if len(ngrambits) > 1:
                ngrambit = ngrambits[1] if len(ngrambits[1]) > len(ngrambits[0]) else ngrambits[0]
            else:
                ngrambit = ngrambits[0]
            ngrambit = ngrambit.strip()
            length = len(ngrambit.split(" "))
            if (length in add) and len(add[length]) < 3 and (ngrambit not in add[length]):
                add[length].append(ngrambit.strip())
            elif (length in add) and len(add[length]) >= 3:
                if len(add[length]) == len(add[4]) == len(add[5]) == len(add[6]) == len(add[7]):
                    print(song)
                    data[song]["ngrams"] = add
                    break
                else:
                    continue
            else:
                continue   
    # Write results to file
    with open("song-info-final.txt", "w+") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def match_tweets(tweets):
    # open csv + pull tweets
    # create copy of tweets array to modify
    '''
    for i in range(len(tweet_copy)):
        if (tweet_copy[i] is in ngrams):
            with open("labeled_data.csv || trump_tweets.csv", "a+") as f:
                f.write(tweets[i] + "\n")
    '''
    # process tweets
    processed_tweets = []
    for tweet in tweets:
        processed_tweet = preprocess(tweet)
        processed_tweet = re.sub(r'&amp;', '&', processed_tweet)
        processed_tweet = processed_tweet.translate(str.maketrans('', '', string.punctuation.replace("-", ""))) 
        processed_tweet = re.sub(r'([Yy]eah)|([Ww]hoah?)|([Oo]h)', '', processed_tweet).strip() #won't remove ooh, remove Ha's
        processed_tweet = processed_tweet.lower()
        processed_tweets.append(processed_tweet)
    # create one big set of ngrams to make matching easier
    ngrams = set()
    with open("song-info-final.txt") as f:
        song_data = json.load(f)
        for song in song_data:
            if (not "ngrams" in song_data[song]):
                continue
            for ngram_len in song_data[song]["ngrams"]:
                for ngram in song_data[song]["ngrams"][ngram_len]:
                    ngrams.add(ngram)
    tweets_to_write = set()
    for i in range(len(processed_tweets)):
        for ngram in ngrams:
            if ngram in processed_tweets[i]:
                tweets_to_write.add(tweets[i])
                print(ngram)
                continue
    with open("trump-tweets-with-lyrics.csv", "w+") as f:
        for tweet in tweets_to_write:
            f.write(tweet + "\n")

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

if __name__ == "__main__":
    # find_songs()
    # find_lyrics()
    # labeled_data.csv and trump_tweets.csv
    # output in same file? 
    if len(sys.argv) != 2:
        print("ERROR: Need one command line argument for a CSV file with tweets to parse.")
        sys.exit(1)
    tweet_file = pd.read_csv(sys.argv[1], 'utf-8', engine="python", names=["text", "date", "fav", "retweets", "id"])
    tweets = tweet_file.text[1:]
    match_tweets(tweets)