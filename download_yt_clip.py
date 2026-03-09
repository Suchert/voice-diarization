import yt_dlp
import sys

def download_video(video_url):
    # Opcje dla yt-dlp, wymuszające pobranie wideo i audio w formacie mp4
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': '%(title)s.%(ext)s', # Nazwa pliku to tytuł filmu
    }

    try:
        print(f"Rozpoczynam pobieranie: {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print("Pobieranie zakończone sukcesem!")
    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania: {e}")

if __name__ == "__main__":
    url = input("Podaj adres URL filmu z YouTube (lub 'q' aby wyjść): ")
    if url.lower() != 'q':
        download_video(url)
