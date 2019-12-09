'''
genius.py utilizes a billboard chart downloader (install with pip install billboard.py) to scrape the past 29 years of top rap/hip-hop
songs as ranked by Billboard, then uses Genius' API to build a dataset of lyrics from the songs.
'''

from requests import get
import billboard
import time
import sys
from requests.exceptions import HTTPError
import lyricsgenius
import nltk
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.collocations import *
from nltk.probability import FreqDist
'''
Current issues: having trouble creating data set due to rate limits on server. Not quite sure what the limits are, so it's challenging.

Issue: timeout means I had to form song list/lyrics in chunks

TODO: Store lyrics (and n-grams) in text files

Return: list of tweets that contain a 4+ word phrase identified as characteristic of song lyrics

'''
def find_songs():
    r_songs = {}
    #start at 2007-03-10
    r_chart = billboard.ChartData('hot-rap-tracks', '2007-03-10')
    while r_chart.previousDate:
        for song in r_chart:
            r_songs[song.title] = song.artist
        try:
            r_chart = billboard.ChartData('hot-rap-tracks', r_chart.previousDate)
            time.sleep(2)
            print(r_chart.previousDate)
            # have a date limit
            if (r_chart.previousDate < "1990-01-01"):
                print("DONE")
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
            print("final (prev) date: " + r_chart.previousDate)
            break
'''
def find_lyrics():

    # Get songs from song-titles.txt --> split by newline, then split by tab
    # keep dictionary of song/artist to track and ignore duplicates
    with open("song-titles.txt") as f:
        data = f.read()
        songs = data.split("\n")

    r_songs["Loyal"] = "Chris Brown"
    genius = lyricsgenius.Genius("xNYwq4ZDn6Ytqu7pJ7Jvy-84TIuWLgwQTDQZ7cMe1ykdDIY7OwSCnblRToxw5N1y")
    r_lyrics = {}
    for song in r_songs:
        print(r_songs[song])
        g_song = genius.search_song(song, r_songs[song])
        r_lyrics[song] = g_song.lyrics

    l_stop_words = ["whoa", "woah", "yeah", "yea", "ah", "ha"]

    # process lyrics
    for song in r_lyrics:
        
        split_lyrics = r_lyrics[song].split("\n")
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
        parsed_lyrics = []
        for i in range(len(new_lyrics)):
            line = new_lyrics[i]
            print(line)
            line = re.sub(r'\([^()]*\)', '', line)
            # REMOVE PUNCTUATION
            line = line.translate(str.maketrans('', '', string.punctuation)) 
            line = re.sub(r'([Yy]eah(ah)?)|([Ww]hoa)|([Oo]+h)', '', line).strip() #won't remove ooh, remove Ha's
            print(line)
            if len(line) != 0:
                parsed_lyrics.append(line)
        all_ngrams = []
        final_lyrics = " asdf ".join(parsed_lyrics)
        vect = CountVectorizer(ngram_range=(5, 8))
        analyzer = vect.build_analyzer()
        listNgramQuery = analyzer(final_lyrics)
        listNgramQuery.reverse()
        #print("listNgramQuery=", listNgramQuery)
        # NgramQueryWeights = nltk.FreqDist(listNgramQuery)
        # print("NgramQueryWeights=", NgramQueryWeights)
        add = {}
        add[4] = []
        add[5] = []
        add[6] = []
        add[7] = []
        count = 0
        for ngram in listNgramQuery:
            print(ngram)
            ngrambits = ngram.split("asdf")
            if len(ngrambits) > 1:
                ngrambit = ngrambits[1] if len(ngrambits[1]) > len(ngrambits[0]) else ngrambits[0]
            else:
                ngrambit = ngrambits[0]
            ngrambit = ngrambit.strip()
            length = len(ngrambit.split(" "))
            if (length in add) and len(add[length]) < 4 and (ngrambit not in add[length]):
                print(ngrambit)
                add[length].append(ngrambit.strip())
            elif (length in add) and len(add[length]) >= 4:
                print(add)
                break
            else:
                continue'''

# calculate the most characteristic lines of the song
# pick top N (n = 2 or 3?) lines and save those
# compare tweets to those top lines

#print(r_lyrics)
# strip blank lines
# strip []
# strip ()

if __name__ == "__main__":
    find_songs()