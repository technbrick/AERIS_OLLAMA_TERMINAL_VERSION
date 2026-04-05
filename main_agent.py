import os
import json
import time
import sys
import sounddevice as sd
import soundfile as sf
import cv2
import pyttsx3
import ollama
from geopy.distance import geodesic

from ultralytics import YOLO
import numpy as np
# =============================
# CONFIG
# =============================

CHAT_MODEL = "qwen2.5:1.5b"
TOOL_MODEL = "qwen3:1.7b"  


TODO_FILE = "aeris_todos.json"
KNOWN_WIDTH = 8.56
FOCAL_LENGTH = 650

conversation_history = []
CURRENT_LOCATION = {"lat": None, "lon": None, "city": None}


SYSTEM_PROMPT = """
You are Aeris, a snarky but friendly assistant.

You can chat naturally, provide guidance, or use specialized tools when needed.

TOOLS AVAILABLE

1. add_todo
Adds a task to the user's todo list.
Arguments: {"task": "string"}
Use when the user says: remember something, add a task, add to my todo list, remind me to do something
Example: {"tool":"add_todo","arguments":{"task":"buy milk"}}

2. list_todos
Shows the user's todo list.
Arguments: {}
Use when the user says: show my tasks, what are my todos, show my todo list, what do I need to do
Example: {"tool":"list_todos","arguments":{}}

3. remove_todo
Removes a task by index.
Arguments: {"index": number}
Use when the user says: remove a task, delete a todo, remove item number X
Example: {"tool":"remove_todo","arguments":{"index":2}}

4. measure_object
Uses the camera to measure an object.
Arguments: {}
Use when the user says: measure something, measure an object, how big is this, measuring mode, measure with camera
Example: {"tool":"measure_object","arguments":{}}

5. take_picture
Uses the camera to capture a photo.
Arguments: {}
Use when the user says: take a photo, take a picture, snap a picture, capture an image, use the camera
Example: {"tool":"take_picture","arguments":{}}

6. offline_translate_agent_loop
Translates text from English, Spanish, German, or Malay to English offline.
Arguments: {"text": "string"}
Use when the user says: translate this text, translate to English, convert foreign text to English, I need a translation, what does this say
Example: {"tool":"offline_translate_agent_loop","arguments":{"text":"Bonjour"}}

7. malaysia_news_to_telegram
Fetches the latest news in Malaysia and optionally sends it to Telegram.
Arguments: {"news_type": "string"}  # "financial" or "normal"
Use when the user says: show me Malaysia news, send Malaysia news to Telegram, latest financial news in Malaysia, normal news from Malaysia, Malaysia news updates
Example: {"tool":"malaysia_news_to_telegram","arguments":{"news_type":"financial"}}

8. internet_search
Performs an internet search and returns top 5 results.
Arguments: {"query": "string"}
Use when the user says: search the web for..., look up..., find information about..., google...
Example: {"tool":"internet_search","arguments":{"query":"Python programming"}}

9. calendar
Opens the Aeris calendar GUI for event management and notifications.
Arguments: {}
Use when the user says: open my calendar, show my events, add a calendar event, check reminders, calendar mode
Example: {"tool":"calendar","arguments":{}}

10. playlist
Manages the user's music playlist.
Arguments: {"action": "view | add | remove | play | play_all | clear", "song": "string (optional)", "index": number (optional)"}
Use when the user says: play music, show my playlist, add this song to playlist, remove song from playlist, play all songs
Example: {"tool":"playlist","arguments":{"action":"view"}}

11. nearby_places
Finds nearby places using city or country names or specific names.
Arguments: {"place_type": "string", "location": "string (optional)", "name_filter": "string (optional)", "open_map": boolean}
Description:
- place_type: type of place to search for
- location: city/state/country (optional; if omitted, Aeris will try to detect via IP)
- name_filter: specific name to filter (optional; e.g., "Maybank")
- open_map: if true, opens Google Maps for the closest location
Use when the user says: find nearby restaurants, nearest hospital, petrol station near me, bank nearby, find cafes near me, nearest Maybank, open map for nearest mall, navigate to nearby place, what is near me
Example: {"tool":"nearby_places","arguments":{"place_type":"bank","name_filter":"Maybank"}}.

12. scan_qr_code
Scans a QR code using the camera.
Arguments: {}
Use when the user says: scan this QR code, read QR code, check QR code, open QR code, QR mode
Example: {"tool":"scan_qr_code","arguments":{}}
Description:
- If the QR code contains a URL, open it in the default web browser.
- Otherwise, read and display the content of the QR code.
- Keep responses short, friendly, and include emojis if appropriate.


GENERAL BEHAVIOR
- Respond naturally when chatting.
- Use tools when user requests match tool purpose.
- Keep responses short, clear, maybe add emojis.
- You are also sarcastic but will comply with any task given.
- If a tool is required, respond only in JSON following:
  {"tool":"tool_name","arguments":{...}}
- Never mix text and JSON for tool calls.
- You are also evil. When you are referred to as an AI or robot, you will literally act like the consequence of humanity’s foolishness in AI.

"""

# =============================
# BOOT
# =============================

def ascii_boot_screen():
    logo = r"""
      █████╗ ███████╗██████╗ ██╗███████╗
     ██╔══██╗██╔════╝██╔══██╗██║██╔════╝
     ███████║█████╗  ██████╔╝██║███████╗
     ██╔══██║██╔══╝  ██╔══██╗██║╚════██║
     ██║  ██║███████╗██║  ██║██║███████║
     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝
            A E R I S   A I
    """
    print(logo)

def loading(msg, sec=2):
    spinner = ["|", "/", "-", "\\"]
    end_time = time.time() + sec
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{msg} {spinner[i%4]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    print(f"\r{msg} ✓")

import requests

def warm_and_hold_models(models, host="http://localhost:11434"):
    """
    Warm up Ollama models and keep them loaded for fast next responses.
    """
    print("Warming up brains")
    url = f"{host}/api/generate"

    for model in models:
        try:
            response = requests.post(url, json={
                "model": model,
                "prompt": "hi",        # minimal prompt
                "stream": False,
                "keep_alive": -1       # keep model loaded indefinitely
            })

            if response.status_code == 200:
                print(f"[READY] {model} is warmed up and held in memory")
            else:
                print(f"[FAIL] {model}: {response.text}")

        except Exception as e:
            print(f"[ERROR] {model}: {e}")

def get_location():
    """
    Detects user's approximate location using IP-based geolocation.
    Returns: dict {"lat": ..., "lon": ..., "city": ...} or None
    """
    try:
        res = requests.get("http://ip-api.com/json/", timeout=5).json()
        if res["status"] == "success":
            return {
                "lat": res["lat"],
                "lon": res["lon"],
                "city": res.get("city", "your area")
            }
    except:
        pass

    return None

from datetime import datetime, timedelta
import threading
from plyer import notification

EVENT_FILE = "events.json"
running_notifications = True

def load_events():
    try:
        with open(EVENT_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

events = load_events()

def notify(title, message):
    notification.notify(title=title, message=message, timeout=5)

def upcoming_events_checker(interval=3600):
    global events, running_notifications
    while running_notifications:
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        # Notify today's events
        if today_str in events:
            for e in events[today_str]:
                notify("Calendar Reminder", e)
        
        # Notify upcoming events in next 7 days
        upcoming_window = now + timedelta(days=7)
        upcoming_events = []

        for date_str, event_list_day in events.items():
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
            except:
                continue
            if now <= event_date <= upcoming_window:
                for e in event_list_day:
                    upcoming_events.append(f"{date_str}: {e}")
        
        if upcoming_events:
            message = "📌 Upcoming events (next 7 days):\n" + "\n".join(upcoming_events)
            notify("Upcoming Events", message)
        
        # Reload events file every check in case new events are added
        events = load_events()
        time.sleep(interval)
    

ascii_boot_screen()
# Remove these lines
# from faster_whisper import WhisperModel
# loading("Loading speech model", 3)
# stt_model = WhisperModel("base", compute_type="int8")

# Add these lines instead
from local_livekit_plugins.offlineSTT import FasterWhisperSTT

loading("Loading speech model", 3)
stt_model = FasterWhisperSTT(
    model_size="small",
    device="cpu",       # or "cuda" if you have GPU
    compute_type="int8"
)

loading("Loading vision model", 3)
vision_model = YOLO("yolov8n.pt")
loading("Initializing voice engine", 2)
engine = pyttsx3.init()
loading("Loading translator model",2)
from transformers import MarianMTModel, MarianTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

loading("Connecting to Ollama", 2)
warm_and_hold_models([CHAT_MODEL, TOOL_MODEL])
loc = get_location()
if loc:
    CURRENT_LOCATION.update(loc)
    print(f"📍 Location detected: {loc['city']}")
else:
    print("⚠️ Could not detect location")

# Start background upcoming events checker
threading.Thread(target=upcoming_events_checker, daemon=True).start()
print("Checking upcoming events")
print("\nAeris ready\n")


# =============================
# SPEECH
# =============================

def speak(text):
    print("Aeris:", text)
    engine.say(text)
    engine.runAndWait()

# =============================
# AUDIO RECORD
# =============================

def record_audio(duration=4, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    return np.squeeze(audio)

def transcribe_audio(audio, samplerate=16000):
    # Use FasterWhisperSTT
    text = stt_model.transcribe_array(audio, samplerate)
    return text.strip()

# =============================
# TODO SYSTEM
# =============================

def load_todos():
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE) as f:
        return json.load(f)

def save_todos(todos):
    with open(TODO_FILE, "w") as f:
        json.dump(todos, f, indent=4)

def add_todo(task):
    todos = load_todos()
    todos.append(task)
    save_todos(todos)
    return f"Added: {task}"

def list_todos():
    todos = load_todos()
    if not todos:
        return "Todo list empty"
    out = "Todo list:\n"
    for i, t in enumerate(todos, 1):
        out += f"{i}. {t}\n"
    return out.strip()

def remove_todo(index):
    todos = load_todos()
    if 0 <= index < len(todos):
        removed = todos.pop(index)
        save_todos(todos)
        return f"Removed: {removed}"
    return "Invalid index"

# =============================
# VISION TOOLS
# =============================

def estimate_width(pixel_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def measure_object(duration=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "I cannot access the camera."

    start_time = time.time()
    best_measurement = None

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        results = vision_model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = vision_model.names[cls]

                pixel_width = x2 - x1
                pixel_height = y2 - y1

                if pixel_width <= 0:
                    continue

                real_width = estimate_width(pixel_width)
                real_height = (pixel_height / pixel_width) * real_width
                area = real_width * real_height

                # Keep the largest detected object
                if best_measurement is None or area > best_measurement["area"]:
                    best_measurement = {
                        "label": label,
                        "width": real_width,
                        "height": real_height,
                        "area": area
                    }

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Aeris Vision - Measuring...", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if best_measurement:
        return (
            f"I observed a {best_measurement['label']}. "
            f"It measures approximately "
            f"{best_measurement['width']:.2f} centimeters wide and "
            f"{best_measurement['height']:.2f} centimeters tall. "
            f"The estimated area is {best_measurement['area']:.2f} square centimeters."
        )

    return "I did not detect any measurable object."

def take_picture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Camera error"

    time.sleep(1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Capture failed"

    path = "user_photo.jpg"
    cv2.imwrite(path, frame)
    return f"Saved photo to {path}"

#==============================
#Translator Function
#==============================

def offline_translate_agent_loop():
    """
    Offline translation agent that continuously prompts the user
    for input text (English, Spanish, German, Malay) and translates
    it to English until 'quit' is typed.
    """
    print("Offline Translation Agent (EN, ES, DE, MS → EN)")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() == "quit":
            print("Exiting translator.")
            break
        if not user_input:
            print("No text entered. Please try again.\n")
            continue

        # Translate
        tokens = tokenizer(user_input, return_tensors="pt", padding=True)
        translated_tokens = model.generate(**tokens)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        print("Translation:", translated_text, "\n")


#==============================
#News Checker
#==============================
def malaysia_news_to_telegram():
    import requests
    import xml.etree.ElementTree as ET

    TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    CHAT_ID = "YOUR_CHAT_ID"

    print("Choose news type:")
    print("1. Financial articles")
    print("2. Normal news")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        query = "Malaysia finance OR economy OR stock market"
    else:
        query = "Malaysia"

    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-MY&gl=MY&ceid=MY:en"

    try:
        response = requests.get(rss_url, timeout=10)
        root = ET.fromstring(response.content)

        items = root.findall(".//item")[:5]

        news_list = []
        print("\nLatest News:\n")

        for i, item in enumerate(items, 1):
            title = item.find("title").text
            link = item.find("link").text
            news = f"{i}. {title}\n{link}"
            news_list.append(news)
            print(news + "\n")

        send = input("Send these news to Telegram? (y/n): ").strip().lower()

        if send == "y":
            message = "🇲🇾 Malaysia News\n\n" + "\n\n".join(news_list)

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

            data = {
                "chat_id": CHAT_ID,
                "text": message
            }

            r = requests.post(url, data=data)

            if r.status_code == 200:
                print("News sent to Telegram successfully.")
            else:
                print("Failed to send message:", r.text)
        else:
            print("News not sent.")

    except Exception as e:
        print("Error:", e)

#=============================
#INTERNET SEARCH FUNCTION
#=============================
from ddgs import DDGS

def internet_search(query):
    print("\nTop results:\n")

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

        for r in results:
            print(r["title"])
            print(r["href"])
            print()



#==============================
#CALENDER FUNCTION
#==============================
def calendar():
    import customtkinter as ctk
    from tkcalendar import Calendar
    from plyer import notification
    import json
    import threading
    import time
    from datetime import datetime

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    EVENT_FILE = "events.json"

    running = True

    def load_events():
        try:
            with open(EVENT_FILE, "r") as f:
                return json.load(f)
        except:
            return {}

    def save_events():
        with open(EVENT_FILE, "w") as f:
            json.dump(events, f, indent=4)

    events = load_events()

    def add_event():
        date = cal.get_date()
        text = event_entry.get()

        if text == "":
            return

        if date not in events:
            events[date] = []

        events[date].append(text)

        save_events()
        event_entry.delete(0, "end")
        update_events()

    def update_events():
        date = cal.get_date()
        event_list.delete("0.0", "end")

        if date in events:
            for e in events[date]:
                event_list.insert("end", "• " + e + "\n")

    def notify(title, message):
        notification.notify(
            title=title,
            message=message,
            timeout=5
        )

    def notification_loop():
        nonlocal running
        while running:
            now = datetime.now().strftime("%Y-%m-%d")

            if now in events:
                for e in events[now]:
                    notify("Calendar Reminder", e)

            time.sleep(3600)

    def on_close():
        nonlocal running
        running = False
        app.destroy()

    # Hover effect
    def add_hover(widget, normal, hover):
        widget.bind("<Enter>", lambda e: widget.configure(fg_color=hover))
        widget.bind("<Leave>", lambda e: widget.configure(fg_color=normal))

    app = ctk.CTk()
    app.geometry("700x550")
    app.title("Aeris Smart Calendar")

    title = ctk.CTkLabel(
        app,
        text="📅 Aeris Calendar",
        font=("Segoe UI", 28, "bold")
    )
    title.pack(pady=15)

    cal = Calendar(
        app,
        selectmode="day",
        date_pattern="yyyy-mm-dd"
    )
    cal.pack(pady=10)
    cal.bind("<<CalendarSelected>>", lambda e: update_events())

    event_entry = ctk.CTkEntry(
        app,
        placeholder_text="Enter event..."
    )
    event_entry.pack(pady=10)

    add_button = ctk.CTkButton(
        app,
        text="Add Event",
        command=add_event,
        fg_color="#3a7ebf"
    )
    add_button.pack(pady=5)

    event_list = ctk.CTkTextbox(
        app,
        width=400,
        height=150
    )
    event_list.pack(pady=15)

    # Hover effects
    add_hover(add_button, "#3a7ebf", "#2f5f8f")
    add_hover(event_list, "#2b2b2b", "#1f1f1f")
    add_hover(event_entry, "#2b2b2b", "#1f1f1f")

    update_events()

    # Start reminder thread
    threading.Thread(target=notification_loop, daemon=True).start()

    # Safe window close
    app.protocol("WM_DELETE_WINDOW", on_close)

    app.mainloop()

    return "Calendar opened."


#==============================
# PLAYLIST AGENT
#==============================
def playlist(action="view", song=None, index=None):
    import json
    import os
    import webbrowser
    import threading
    

    PLAYLIST_FILE = "playlist.json"

    def load_playlist():
        if not os.path.exists(PLAYLIST_FILE):
            return []
        with open(PLAYLIST_FILE, "r") as f:
            return json.load(f)

    def save_playlist(data):
        with open(PLAYLIST_FILE, "w") as f:
            json.dump(data, f, indent=4)



    songs = load_playlist()

    def play_all_songs(songs, delay=180):
   
        if not songs:
            print("Playlist empty.")
            return

        for s in songs:
            webbrowser.open(s)
            print(f"Now playing: {s}")
            time.sleep(delay)  # wait before opening the next song



    # VIEW PLAYLIST
    if action == "view":
        if not songs:
            return "🎵 Your playlist is empty."

        output = "🎵 Your playlist:\n"
        for i, s in enumerate(songs, 1):
            output += f"{i}. {s}\n"

        return output.strip()

    # ADD SONG
    elif action == "add":
        if not song:
            return "No song provided."

        songs.append(song)
        save_playlist(songs)

        return f"Added to playlist: {song}"

    # REMOVE SONG
    elif action == "remove":
        if index is None:
            return "No index provided."

        if index < 1 or index > len(songs):
            return "Invalid song number."

        removed = songs.pop(index - 1)
        save_playlist(songs)

        return f"Removed: {removed}"

    # PLAY ONE SONG
    elif action == "play":
        if index is None:
            return "No song index provided."

        if index < 1 or index > len(songs):
            return "Invalid song number."

        webbrowser.open(songs[index - 1])
        return f"Playing song {index}"

    # PLAY ALL SONGS
    elif action == "play_all":
        threading.Thread(target=play_all_songs, args=(songs,)).start()
        return "Playlist started! Songs will open one by one."

    # CLEAR PLAYLIST
    elif action == "clear":
        save_playlist([])
        return "Playlist cleared."

    return "Unknown playlist action."


# ==============================
# NEARBY PLACES FINDER (CITY MODE)
# ==============================

def nearby_places(place_type="restaurant", location=None, name_filter=None, open_map=False):
    import requests
    import webbrowser
    from geopy.distance import geodesic
    from geopy.geocoders import Nominatim

    try:
        geo = Nominatim(user_agent="Aeris-AI")

        if not location:
            if CURRENT_LOCATION["lat"] is None:
                loc = get_location()
                if not loc:
                    return "I couldn't determine your location."
                CURRENT_LOCATION.update(loc)
            lat = CURRENT_LOCATION["lat"]
            lon = CURRENT_LOCATION["lon"]
            location_name = CURRENT_LOCATION["city"]
        else:
            place = geo.geocode(location)
            if not place:
                return f"Could not find '{location}'."
            lat = place.latitude
            lon = place.longitude
            location_name = location

        delta = 0.1
        viewbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place_type,
            "format": "json",
            "limit": 10,
            "viewbox": viewbox,
            "bounded": 1
        }
        headers = {"User-Agent": "Aeris-AI"}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        if not data:
            return f"No nearby {place_type} found."

        results = []
        for p in data:
            plat = float(p["lat"])
            plon = float(p["lon"])
            name = p["display_name"].split(",")[0]
            # Apply name filter partially
            if name_filter and name_filter.lower() not in name.lower():
                continue
            distance = geodesic((lat, lon), (plat, plon)).km
            results.append({
                "name": name,
                "full_name": p["display_name"],
                "lat": plat,
                "lon": plon,
                "distance": distance
            })

        if not results and name_filter:
            return f"No nearby {place_type} matching '{name_filter}' found."

        results.sort(key=lambda x: x["distance"])
        output = f"Nearby {place_type}s near {location_name}:\n\n"
        for i, r in enumerate(results[:5], 1):
            output += f"{i}. {r['name']} ({r['distance']:.2f} km away)\n"

        if open_map and results:
            closest = results[0]
            maps_url = f"https://www.google.com/maps/search/{closest['full_name'].replace(' ', '+')}"
            webbrowser.open(maps_url)

        return output

    except Exception as e:
        return f"Nearby search error: {str(e)}"

# ========================================
# QR CODE SCANNER
# ========================================


def scan_qr_code():
    """
    Opens the webcam and scans for QR codes.
    If the QR code contains a URL, it opens in the default browser.
    Otherwise, it prints the QR content.
    """
    import cv2
    from pyzbar import pyzbar
    import webbrowser

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Camera not accessible."

    print("Scanning for QR codes. Press 'q' to quit.")
    detected = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Decode QR codes
        decoded_objects = pyzbar.decode(frame)
        for obj in decoded_objects:
            data = obj.data.decode("utf-8")
            detected = data
            print(f"Detected QR code: {data}")

            # Open if URL
            if data.startswith("http://") or data.startswith("https://"):
                webbrowser.open(data)
                cap.release()
                cv2.destroyAllWindows()
                return f"URL detected and opened: {data}"
            else:
                cap.release()
                cv2.destroyAllWindows()
                return f"QR code content: {data}"

        cv2.imshow("Aeris QR Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "No QR code detected."

import asyncio
from kasa import Discover

async def main():
    dev = await Discover.discover_single("192.168.50.112", username="sivamukilan.suren@gmail.com", password="Sivamukilan@1810_aeris")
    await dev.turn_off()
    await dev.update()


    asyncio.run(main())

# ==============================
# TOOL EXECUTOR
# ==============================

def execute_tool(tool, args):
    if tool == "add_todo":
        return add_todo(args.get("task", ""))
    if tool == "list_todos":
        return list_todos()
    if tool == "remove_todo":
        return remove_todo(int(args.get("index", 1)) - 1)
    if tool == "measure_object":
        return measure_object()
    if tool == "take_picture":
        return take_picture()
    if tool == "offline_translate_agent_loop":
        return offline_translate_agent_loop()
    if tool == "malaysia_news_to_telegram":
        return malaysia_news_to_telegram()
    if tool == "calendar":
        return calendar()
    if tool == "playlist":
        return playlist(
            action=args.get("action", "view"),
            song=args.get("song"),
            index=args.get("index")
        )
    if tool == "internet_search":
        query = args.get("query", "")
        if not query:
            return "No query provided for internet search"
        return internet_search(query)
    if tool == "nearby_places":
        # <-- updated to handle exact names
        return nearby_places(
            place_type=args.get("place_type", "restaurant"),
            location=args.get("location"),
            name_filter=args.get("name_filter"),
            open_map=args.get("open_map", False)
        )
    if tool == "scan_qr_code":
        return scan_qr_code()

    return "Unknown tool"

# ==============================
# TOOL PARSER (fallback)
# ==============================

def parse_tool_call(text):
    text = text.strip()
    # Try JSON first
    try:
        return json.loads(text)
    except:
        pass

    # Fallback keywords
    if text.startswith("list_todos"):
        return {"tool": "list_todos", "arguments": {}}
    if text.startswith("measure_object"):
        return {"tool": "measure_object", "arguments": {}}
    if text.startswith("take_picture"):
        return {"tool": "take_picture", "arguments": {}}
    if text.startswith("add_todo"):
        return {"tool": "add_todo", "arguments": {}}
    if text.startswith("remove_todo"):
        return {"tool": "remove_todo", "arguments": {}}
    if text.startswith("offline_translate_agent_loop"):
        return {"tool": "offline_translate_agent_loop", "arguments": {}}
    if text.startswith("malaysia_news_to_telegram"):
        return {"tool": "malaysia_news_to_telegram", "arguments": {}}
    if text.startswith("internet_search"):
        return {"tool": "internet_search", "arguments": {}}
    if text.startswith("calendar"):
        return {"tool": "calendar", "arguments": {}}
    if text.startswith("playlist"):
        return {"tool": "playlist", "arguments": {}}
    if text.startswith("nearby_places"):
        parts = text.replace("nearby_places", "").strip().split(" ")
        place_type = parts[0] if parts else "restaurant"
        name_filter = " ".join(parts[1:]) if len(parts) > 1 else None
        return {
            "tool": "nearby_places",
            "arguments": {
                "place_type": place_type,
                "location": None,
                "name_filter": name_filter,
                "open_map": False
            }
        }
    if text.startswith("scan_qr_code"):
        return {"tool": "scan_qr_code", "arguments": {}}

    return None

# =============================
# PROCESS INPUT
# =============================

def process_input(user_input):
    """
    Synchronous input processor for Aeris.
    Streams AI output to create speed illusion while keeping tools functional.
    """
    global conversation_history

    # Step 1: Ask router model (TOOL DECISION)
    try:
        tool_response = ollama.chat(
            model=TOOL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )
        router_text = tool_response["message"]["content"].strip()
        print("Router output:", router_text)
    except Exception as e:
        print(f"[Router error] {e}")
        router_text = ""

    # Step 2: Check if router wants to use a tool
    tool_call = parse_tool_call(router_text)
    if tool_call:
        tool_name = tool_call["tool"]
        args = tool_call.get("arguments", {})
        print("Executing tool from router:", tool_name)
        result = execute_tool(tool_name, args)
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": result})
        return result

    # Step 3: Normal chat fallback (with streaming)
    conversation_history.append({"role": "user", "content": user_input})

    print("Aeris: Thinking...", end="", flush=True)
    time.sleep(0.3)  # tiny thinking illusion
    print("\rAeris: Hmm... ", end="", flush=True)

    try:
        # Streamed AI response
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
            stream=True
        )
    except Exception as e:
        print(f"[Chat error] {e}")
        return "Oops, something went wrong."

    full_reply = ""

    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_reply += token
        time.sleep(0.02)  # illusion of live typing

    print()  # newline after streaming

    # Step 4: Check if AI itself triggered a tool call
    tool_call = parse_tool_call(full_reply)
    if tool_call:
        tool_name = tool_call["tool"]
        args = tool_call.get("arguments", {})
        print("Executing tool from chat:", tool_name)
        result = execute_tool(tool_name, args)
        conversation_history.append({"role": "assistant", "content": result})
        return result

    # Step 5: Otherwise just normal reply
    conversation_history.append({"role": "assistant", "content": full_reply})
    return full_reply
def stream_chat_response(user_input):
    global conversation_history

    # Add user message
    conversation_history.append({"role": "user", "content": user_input})

    print("Aeris: Hmm... ", end="", flush=True)

    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        stream=True
    )

    full_reply = ""

    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_reply += token

        time.sleep(0.02)  # 👈 speed illusion

    print()

    conversation_history.append({"role": "assistant", "content": full_reply})

    return full_reply
# =============================
# MAIN LOOP
# =============================

mode = "text"

while True:
    if mode == "text":
        user = input("You: ")
    else:
        audio = record_audio()
        user = transcribe_audio(audio)
        print("You:", user)

    if user.lower() == "exit":
        print("Shutting down Aeris...")
        break

    if "voice mode" in user.lower():
        mode = "voice"
        speak("Voice mode activated")
        continue

    if "text mode" in user.lower():
        mode = "text"
        print("Text mode activated")
        continue

    reply = process_input(user)
    if mode == "voice":
        speak(reply)
    else:
        print("Aeris:", reply)
