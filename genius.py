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
'''
Current issues: having trouble creating data set due to rate limits on server. Not quite sure what the limits are, so it's challenging.
TODO: pick chart for rap songs, chart for hip-hop songs
TODO: store data in json

Potential issues: lyricsgenius may not recognzie the form of the song titles from Genius --> possible fix, remove parentheticals

Return: list of tweets that contain a 4+ word phrase identified as characteristic of song lyrics

'''
r_songs = {}
'''
r_chart = billboard.ChartData('hot-rap-tracks')
count = 0
while r_chart.previousDate:
    for song in r_chart:
        r_songs[song.title] = song.artist
    try:
        r_chart = billboard.ChartData('hot-rap-tracks', r_chart.previousDate)
        count = count + 1
        time.sleep(2)
        print(r_chart.previousDate)
        # have a date limit
        if (r_chart.previousDate < "1990-01-01"):
            break
    except HTTPError as err:
        print("Error reached, waiting 2 minutes to try again...")
        time.sleep(120)
        print("current song list size: " + str(len(r_songs)))
'''
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
    for i in range(len(new_lyrics)):
        line = new_lyrics[i]
        print(line)
        line = re.sub(r'\([^()]*\)', '', line)
        # REMOVE PUNCTUATION
        line = line.translate(str.maketrans('', '', string.punctuation)) 
        line = re.sub(r'([Yy]eah(ah)?)|([Ww]hoa)|([Oo]+h)', '', line).strip() #won't remove ooh, remove Ha's
        print(line)

# use PMI to calculate the most characteristic lines of the song
# pick top N (n = 2 or 3?) lines and save those
# compare tweets to those top lines

#print(r_lyrics)
# strip blank lines
# strip []
# strip ()
