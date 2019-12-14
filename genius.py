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
    data = {}
    # Get songs from song-titles.txt --> split by newline, then split by tab
    # keep dictionary of song/artist to track and ignore duplicates
    with open("song-titles.txt") as f:
        s_data = f.read()
        songs = s_data.split("\n")
        print(songs)
    for song in songs:
        info = song.split("\t")
        if info[0] == "Sunflower (Spider-Man: Into The Spider-Verse)":
            info[0] = "Sunflower"
        data[info[0]] = {}
        data[info[0]]["author"] = info[1]
    #sys.exit()
    genius = lyricsgenius.Genius("xNYwq4ZDn6Ytqu7pJ7Jvy-84TIuWLgwQTDQZ7cMe1ykdDIY7OwSCnblRToxw5N1y")
    lyrics = {}
    for song in data:
        try: 
            g_song = genius.search_song(song, data[song]["author"])
            if g_song is None:
                print("yikes! there's an issue with this one")
                with open("song-title-issues.txt", "a+") as f:
                    f.write(song + "\n")
                continue
            lyrics[song] = g_song.lyrics
        except: 
            print("yikes! there's an issue with this one")
            with open("song-title-issues.txt", "a+") as f:
                f.write(song + "\n")
            continue

    lyric_stop_words = ["whoa", "woah", "yeah", "yea", "ah", "ha"]

    # Process lyrics
    for song in lyrics:
        split_lyrics = lyrics[song].split("\n")
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
    with open("song-info.txt", "w+") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    for song in data:
        '''
        song_lyrics = 
        parsed_lyrics = []
        for i in range(len(new_lyrics)):
            line = new_lyrics[i]
            print(line)
            line = re.sub(r'\([^()]*\)', '', line)
            # REMOVE PUNCTUATION
            line = line.translate(str.maketrans('', '', string.punctuation)) 
            # line = re.sub(r'([Yy]eah(ah)?)|([Ww]hoah?|(([Oo])+h)', '', line).strip() #won't remove ooh, remove Ha's
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
                continue
        '''    
'''
# calculate the most characteristic lines of the song
# pick top N (n = 2 or 3?) lines and save those
# compare tweets to those top lines

#print(r_lyrics)
# strip blank lines
# strip []
# strip ()
'''
if __name__ == "__main__":
    #with open("song-titles.txt", 'a+') as f:
    #    f.write("Gimme Some More\tBusta Rhymes")
    find_lyrics()
    #r_chart = billboard.ChartData('hot-rap-tracks', '1999-02-20')
    #print()