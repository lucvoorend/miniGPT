import requests
from bs4 import BeautifulSoup

# Function to fetch and parse a webpage
def get_soup(url):
    response = requests.get(url)
    return BeautifulSoup(response.text, 'html.parser')

# URL of the Acda en de Munnik song lyrics list
base_url = "https://songteksten.net"
artist_url1 = f"{base_url}/artist/lyrics/11/acda-en-de-munnik.html"
artist_url2 = f"{base_url}/artist/lyrics/11/acda-en-de-munnik.html?page=2"
artist_url3 = f"{base_url}/artist/lyrics/11/acda-en-de-munnik.html?page=3"

# Function to scrape song lyrics from a given song link
def scrape_lyrics(song_url):
    soup = get_soup(song_url)
    # Find the div containing the lyrics (no class, but first sub div in "uk-article" class)
    lyrics = soup.find('article', class_='uk-article')
    if lyrics:
        lyrics = lyrics.get_text()
        # Remove the last lines (not actual lyrics)
        lyrics = lyrics.split('\n')[:-15]

        # Remove any accented characters and replace with closest ASCII equivalent
        lyrics = [lyric.encode('ascii', 'ignore').decode() for lyric in lyrics]
        return '\n'.join(lyrics)
    else:
        return None


# Function to scrape all song links from the main artist page
def scrape_song_links(artist_url):
    soup = get_soup(artist_url)
    song_links = []
    for song in soup.find_all('a', href=True):
        if 'lyric' in song['href']:  # Filter for actual song links
            song_links.append(f"{song['href']}")
    # Remove first 7 elements (not actual song links)
    song_links =  song_links[7:]
    # Remove last 7 elements (not actual song links)
    song_links = song_links[:-7]
    return song_links

# Main scraping process
def scrape_acda_en_de_munnik():
    total_song_links = []
    for url in [artist_url1, artist_url2, artist_url3]:
        artist_url = url
        song_links = scrape_song_links(artist_url)
        total_song_links.extend(song_links)

    print(f"Found {len(total_song_links)} song links")

    # Scrape lyrics and write to a txt file
    with open('acda_en_de_munnik_lyrics.txt', 'w') as f:
        for song in total_song_links:
            lyrics = scrape_lyrics(song)
            if lyrics:
                f.write(lyrics)
                f.write('\n\n')
                print(f"Scraped lyrics for {song}")
    
    print("Finished scraping")

if __name__ == '__main__':
    scrape_acda_en_de_munnik()