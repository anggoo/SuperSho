import os
import streamlit as st
from datetime import datetime,timedelta, date
from PIL import Image
import requests
import toml
from geopy.distance import geodesic
from streamlit_js_eval import get_geolocation
import socket
import uuid
import platform
from geopy.geocoders import Nominatim

# Load configuration from secrets.toml
with open("secrets.toml", "r") as f:
    config = toml.load(f)

# Telegram
BOT_TOKEN = config["BOT_TOKEN"]
CHAT_IDS = config["TELEGRAM_CHAT_IDS"]
latitude = config["latitude"]
longitude = config["longitude"]


# Hardcoded user list
USERS = [
    "sham",
    "pragya",
    "saloni",
    "shomaila"
]

# Allowed location (latitude, longitude)
ALLOWED_LOCATION = (latitude, longitude)  # Change as needed
ALLOWED_RADIUS_KM = 1.0  # Allowed radius in kilometers

def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)

def get_device_info():
    hostname = socket.gethostname()
    mac = ':'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff)
                    for ele in range(0,8*6,8)][::-1])
    user = os.getlogin() if hasattr(os, "getlogin") else "Unknown"
    system = platform.system()
    release = platform.release()
    return f"Hostname: {hostname}\nMAC: {mac}\nOS User: {user}\nSystem: {system} {release}"

def is_within_allowed_location(lat, lon):
    user_location = (latitude, longitude)
    distance = geodesic(user_location, ALLOWED_LOCATION).kilometers
    return distance <= ALLOWED_RADIUS_KM

st.title("Attendance Submission")

# Get user's geolocation
location = get_geolocation()
if location is None:
    st.warning("Waiting for location data...")
else:
    lat = location['coords']['latitude']
    lon = location['coords']['longitude']
    if not is_within_allowed_location(lat, lon):
        st.error("You are not within the allowed location to log your attendance.")
    else:
        user = st.selectbox("Select your name", USERS)
        submit = st.button("Submit Attendance")
        if submit and user:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            device_info = get_device_info()
            message = (
                f"Attendance submitted:\n"
                f"User: {user}\n"
                f"Time: {now}\n"
                f"Location: ({lat}, {lon})\n"
                f"{device_info}"
            )
            send_to_telegram(message)
            st.success("Attendance submitted and sent to the manager!")

            st.header("Leave Submission")

leave_type = st.selectbox("Leave Type", ["Sick Leave", "Personal Leave", "Annual Leave"])
today = date.today()

if leave_type == "Personal Leave":
    leave_date = st.date_input("Select leave date", value=today, min_value=today, max_value=None)
elif leave_type == "Sick Leave":
    leave_date = st.date_input("Select leave date", value=today, min_value=today, max_value= today)
elif leave_type == "Annual Leave":
    leave_start = st.date_input(
        "Select start date", value=today + timedelta(days=7), min_value=today + timedelta(days=7), max_value=None
    )
    leave_end = st.date_input(
        "Select end date", value=leave_start, min_value=leave_start, max_value=None
    )
leave_submit = st.button("Submit Leave")

if leave_submit:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if leave_type == "Annual Leave":
        leave_info = f"Leave Dates: {leave_start} to {leave_end}"
    else:
        leave_info = f"Leave Date: {leave_date}"
    message = (
        f"Leave Submitted:\n"
        f"User: {user}\n"
        f"Type: {leave_type}\n"
        f"{leave_info}\n"
        f"Submission Time: {now}"
    )
    send_to_telegram(message)
    st.success("Leave submitted and sent to the manager!")

st.info("No login required. No data is stored locally.")
