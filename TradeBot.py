import cv2
import numpy as np
import pyautogui
import time
import logging
import json
from random import randint
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import threading
import sys
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from skimage import transform

# Initialize logging for error tracking
logging.basicConfig(filename='tradebot_errors.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

state_file = 'trade_state.json'
item_counts_file = r"item_counts.json"  # JSON file updated by Item Quantity Counter
activity_timeout = timedelta(seconds=240)  # 4 minutes
last_activity_time = datetime.now()

# Coordinates for the trade slots
your_slot_coords = (1640, 497, 21, 20)
their_slot_coords = (1306, 498, 21, 20)
blank_slot_coords = (1287, 528, 61, 60)

# Inventory dimensions
inventory_coords = (2029, 225, 288, 220)

# Create a directory to save the screenshots
output_dir = r"images"
os.makedirs(output_dir, exist_ok=True)

# Path to the blank reference image
blank_image_path = r"Blank.png"

# Item templates mapping
item_templates = {
    "dino_egg": ("dino_egg_inventory.png", "dino_egg.png"),
    "icecream_machine": ("icm_inventory.png", "icm.png"),
    "majestic_chair": ("MJS_INVENTORY.png", "MJS.png"),
    "petal_patch": ("petal_inventory.png", "petal.png"),
    "purple_pillow": ("pillow_inventory.png", "Pillow.png"),
    "hc_sofa": ("Sofa_inv.png", "Sofa.png"),
    "cola_machine": ("Cola_inventory.png", "cola.png"),
}

# Pushover credentials (replace with your credentials)
pushover_user_key = "your_pushover_user_key"
pushover_api_token = "your_pushover_api_token"

def send_pushover_notification(message):
    try:
        r = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": pushover_api_token,
            "user": pushover_user_key,
            "message": message,
        })
        if r.status_code != 200:
            logging.error(f"Failed to send Pushover notification: {r.text}")
        else:
            logging.info("Pushover notification sent successfully.")
    except Exception as e:
        logging.error(f"Error sending Pushover notification: {str(e)}")
        logging.error(traceback.format_exc())

def save_slot_screenshot(screen, slot_coords, label, timestamp):
    x, y, w, h = slot_coords
    slot_region = screen[y:y + h, x:x + w]
    filename = os.path.join(output_dir, f'{label}_{timestamp}.png')
    cv2.imwrite(filename, slot_region)
    logging.info(f"Saved {label} screenshot: {filename}")

# Save state to file
def save_state(current_message):
    state = {
        'current_message': current_message
    }
    with open(state_file, 'w') as f:
        json.dump(state, f)

# Load state from file
def load_state():
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            return state.get('current_message', "Selling Dino for Cola")
    except FileNotFoundError:
        return "Selling Dino for Cola"

# Test image recognition
def test_image_recognition():
    try:
        trade_window = pyautogui.locateOnScreen('images/trade_window.png', confidence=0.8)
        if trade_window:
            print("Trade window found.")
        else:
            print("Trade window not found.")

        dino_egg = pyautogui.locateOnScreen('images/dino_egg_inventory.png', confidence=0.8)
        if dino_egg:
            print("Dino egg found.")
        else:
            print("Dino egg not found.")

        cola = pyautogui.locateOnScreen('images/cola.png', confidence=0.8)
        if cola:
            print("Cola found.")
        else:
            print("Cola not found.")

        my_item_box = pyautogui.locateOnScreen('images/my_item_box.png', confidence=0.8)
        if my_item_box:
            print("My item box found.")
        else:
            print("My item box not found.")

        accept_button = pyautogui.locateOnScreen('images/accept_button.png', confidence=0.8)
        if accept_button:
            print("Accept button found.")
        else:
            print("Accept button not found.")

        their_item_box = pyautogui.locateOnScreen('images/their_item_box.png', confidence=0.8)
        if their_item_box:
            print("Their item box found.")
        else:
            print("Their item box not found.")

        inventory_with_eggs = pyautogui.locateOnScreen('images/Inventory_with_eggs.png', confidence=0.8)
        if inventory_with_eggs:
            print("Inventory with eggs found.")
        else:
            print("Inventory with eggs not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Get the number of items in inventory using OpenCV and NumPy
def get_item_count_and_positions(template_image_path):
    try:
        screenshot = pyautogui.screenshot(region=inventory_coords)
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        cv2.imwrite('screenshot_debug.png', screenshot)

        template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Failed to load {template_image_path}")
            logging.error(f"Failed to load {template_image_path}")
            return 0, []

        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('screenshot_gray_debug.png', screenshot_gray)

        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where(result >= threshold)

        print(f"Template matching result: {result}")
        logging.info(f"Template matching result: {result}")
        print(f"Locations found: {list(zip(*loc[::-1]))}")
        logging.info(f"Locations found: {list(zip(*loc[::-1]))}")

        min_distance = 10
        filtered_points = []
        for point in zip(*loc[::-1]):
            if all(np.linalg.norm(np.array(point) - np.array(fp)) >= min_distance for fp in filtered_points):
                filtered_points.append(point)

        count = len(filtered_points)
        adjusted_points = [(point[0] + inventory_coords[0] + template.shape[1] // 2,
                            point[1] + inventory_coords[1] + template.shape[0] // 2) for point in filtered_points]

        logging.info(f"{template_image_path} count: {count}")
        print(f"{template_image_path} count: {count}")
        return count, adjusted_points
    except Exception as e:
        logging.error(f"Error getting {template_image_path} count: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Error getting {template_image_path} count: {str(e)}")
        return 0, []

# Preprocess the image for better template matching accuracy
def preprocess_image(image):
    if len(image.shape) == 2:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Load templates for number recognition
def load_templates():
    templates_me = {}
    templates_other = {}
    for label in range(0, 10):  # Load templates for digits 0-9
        me_template_path = os.path.join(output_dir, f'Me/{label}.png')
        other_template_path = os.path.join(output_dir, f'Other/Other_{label}.png')
        empty_template_path = os.path.join(output_dir, f'Empty.png')
        if os.path.exists(me_template_path):
            templates_me[f'Me_{label}'] = cv2.imread(me_template_path, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(other_template_path):
            templates_other[f'Other_{label}'] = cv2.imread(other_template_path, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(empty_template_path):
            templates_me['Empty'] = cv2.imread(empty_template_path, cv2.IMREAD_GRAYSCALE)
            templates_other['Empty'] = cv2.imread(empty_template_path, cv2.IMREAD_GRAYSCALE)
    return templates_me, templates_other

# Match template
def match_template(image, templates):
    best_match = None
    max_val = -np.inf
    for label, template in templates.items():
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            logging.debug(f"Template {label} is larger than the image region.")
            continue
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_loc_val, _, _ = cv2.minMaxLoc(result)
        if max_loc_val > max_val:
            max_val = max_loc_val
            best_match = label
    logging.info(f"Best match: {best_match} with value: {max_val}")
    return best_match

# Check if the slot image is blank
def is_blank_image(image, blank_image_path):
    gray_image = preprocess_image(image)
    blank_image = preprocess_image(cv2.imread(blank_image_path))

    if gray_image.shape != blank_image.shape:
        blank_image = cv2.resize(blank_image, (gray_image.shape[1], gray_image.shape[0]))

    is_blank = np.array_equal(gray_image, blank_image)
    logging.info(f"Is blank image: {is_blank}")
    return is_blank

# Read number from trade slot
def read_number_from_trade_slot(screen, slot_coords, templates, label):
    x, y, w, h = slot_coords
    slot_region = screen[y:y + h, x:x + w]
    gray_slot = preprocess_image(slot_region)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_slot_screenshot(screen, slot_coords, label, timestamp)
    if is_blank_image(slot_region, blank_image_path):
        return 0
    matched_number = match_template(gray_slot, templates)
    if matched_number is None or matched_number == 'Empty':
        return 0
    return int(matched_number.split('_')[-1])

# Analyze the screenshot
def analyze_screenshot(templates):
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    your_item_count = read_number_from_trade_slot(screenshot, your_slot_coords, templates, 'Your_Item')
    print(f"MY item COUNT: {your_item_count}")

    their_item_count = read_number_from_trade_slot(screenshot, their_slot_coords, templates, 'Their_Item')
    print(f"OTHER players item COUNT: {their_item_count}")

    cv2.imwrite(os.path.join(output_dir, "screenshot.png"), screenshot)

# Count items in the trade slot using template matching
def count_items_in_trade_slot(templates_me, templates_other):
    try:
        slot_regions = [
            (377, 173, 50, 20),
            (488, 173, 50, 20),
            (600, 173, 50, 20),
            (377, 278, 50, 20),
            (488, 278, 50, 20),
            (600, 278, 50, 20),
            (813, 173, 50, 20),
            (925, 173, 50, 20),
            (1037, 173, 50, 20),
            (813, 278, 50, 20),
            (925, 278, 50, 20),
            (1037, 278, 50, 20)
        ]

        item_counts = []
        for i, region in enumerate(slot_regions):
            trade_slot_image = pyautogui.screenshot(region=region)
            trade_slot_image = np.array(trade_slot_image)
            gray_slot = preprocess_image(trade_slot_image)
            if i < 6:
                matched_number = match_template(gray_slot, templates_other)
            else:
                matched_number = match_template(gray_slot, templates_me)

            if matched_number is None or matched_number == 'Empty':
                item_counts.append(0)
            else:
                item_counts.append(int(matched_number.split('_')[-1]))

            logging.info(f"Items in trade slot {region}: {item_counts[-1]}")
        return item_counts

    except Exception as e:
        logging.error(f"Error counting items in trade slot: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Error counting items in trade slot: {str(e)}")
        return [0] * 12

# Check for activity on the other player's trade screen
def check_trade_activity(templates_me, templates_other):
    global last_activity_time
    current_state = count_items_in_trade_slot(templates_me, templates_other)[:6]
    if any(count > 0 for count in current_state):
        last_activity_time = datetime.now()
    return current_state

# Cancel the trade if there is no activity for a specified duration
def cancel_trade_if_inactive():
    global last_activity_time
    if datetime.now() - last_activity_time > activity_timeout:
        cancel_button = pyautogui.locateOnScreen('images/cancel_button.png', confidence=0.8)
        if cancel_button:
            print("Cancel button found, cancelling trade due to inactivity.")
            pyautogui.click(cancel_button)
        last_activity_time = datetime.now()

# Click the next page button in the inventory
def click_next_page():
    next_page_button = pyautogui.locateOnScreen('images/next_page_button.png', confidence=0.8)
    if next_page_button:
        pyautogui.click(next_page_button)
        time.sleep(7)
        print("Next page button clicked.")
        logging.info("Next page button clicked.")
        return True
    else:
        print("Next page button not found.")
        logging.info("Next page button not found.")
        return False

# Complete the trade
def complete_trade(my_item_image, their_item_image, item_count, want_item_count, templates_me, templates_other):
    try:
        print("Completing trade...")

        their_item = pyautogui.locateOnScreen(f'images/Items/{their_item_image}', confidence=0.8)
        if their_item:
            print(f"Their item {their_item_image} found.")
        else:
            print(f"Their item {their_item_image} not found.")
            return

        while True:
            with open(item_counts_file, 'r') as f:
                item_counts = json.load(f)
            their_item_count = item_counts.get("their_item_count", 0)
            if their_item_count >= want_item_count:
                break
            time.sleep(1)
            cancel_trade_if_inactive()

        items_to_add = item_count
        page_swaps = 0
        while items_to_add > 0:
            count, positions = get_item_count_and_positions(f'images/Items/{my_item_image}')
            if count == 0:
                if click_next_page():
                    page_swaps += 1
                if page_swaps >= 10:
                    send_pushover_notification(f"Failed to find {my_item_image} in inventory after 10 page swaps. Cancelling trade.")
                    cancel_button = pyautogui.locateOnScreen('images/cancel_button.png', confidence=0.8)
                    if cancel_button:
                        pyautogui.click(cancel_button)
                    return
                continue

            for position in positions[:items_to_add]:
                pyautogui.moveTo(position)
                time.sleep(0.5)
                pyautogui.click(position)
                time.sleep(0.5)
                my_item_box = pyautogui.locateOnScreen('images/my_item_box.png', confidence=0.8)
                if my_item_box:
                    pyautogui.moveTo(my_item_box)
                    time.sleep(0.5)
                    pyautogui.click()
                    time.sleep(0.5)
                else:
                    return
                items_to_add -= 1

        while True:
            with open(item_counts_file, 'r') as f:
                item_counts = json.load(f)
            your_item_count = item_counts.get("your_item_count", 0)
            if your_item_count >= item_count:
                break
            time.sleep(1)
            cancel_trade_if_inactive()

        accept_button = pyautogui.locateOnScreen('images/accept_button.png', confidence=0.8)
        if accept_button:
            pyautogui.click(accept_button)
            time.sleep(60)
            with open(item_counts_file, 'r') as f:
                item_counts = json.load(f)
            if item_counts.get("their_item_count", 0) < want_item_count:
                cancel_button = pyautogui.locateOnScreen('images/cancel_button.png', confidence=0.8)
                if cancel_button:
                    pyautogui.click(cancel_button)
        else:
            print("Accept button not found.")
    except Exception as e:
        logging.error(f'Failed to complete trade: {str(e)}')
        print(f"An error occurred while completing trade: {str(e)}")

# Perform the trade and spamming
def perform_trade(item, item_count, want_item, want_item_count, stop_event):
    global last_activity_time
    try:
        current_message = load_state()
        trade_message_1 = f"SELL {item_count} {item.upper()} FOR {want_item_count} {want_item.upper()}"
        trade_message_2 = f"SELL {want_item_count} {want_item.upper()} FOR {item_count} {item.upper()}"

        templates_me, templates_other = load_templates()

        while not stop_event.is_set():
            current_message = f"SELL {item_count} {item.upper()} FOR {want_item_count} {want_item.upper()}"

            pyautogui.write(current_message)
            pyautogui.press('enter')
            time.sleep(randint(5, 10))

            if is_trade_open():
                print("Trade window is open, starting trade process...")

                screenshot = pyautogui.screenshot()
                screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                if is_blank_image(screenshot[blank_slot_coords[1]:blank_slot_coords[1] + blank_slot_coords[3], blank_slot_coords[0]:blank_slot_coords[0] + blank_slot_coords[2]], blank_image_path):
                    print("The trade slot is blank.")
                else:
                    print("The trade slot is not blank.")
                    if current_message == trade_message_1:
                        complete_trade(item_templates[item][0], item_templates[want_item][1], item_count, want_item_count, templates_me, templates_other)
                        current_message = trade_message_2
                    else:
                        complete_trade(item_templates[want_item][0], item_templates[item][1], want_item_count, item_count, templates_me, templates_other)
                        current_message = trade_message_1

                save_state(current_message)

            check_trade_activity(templates_me, templates_other)
            cancel_trade_if_inactive()

    except Exception as e:
        logging.error(f'Bot encountered an error: {str(e)}')
        print(f'Bot encountered an error: {str(e)}')

# Function to check if trade has started
def is_trade_open():
    try:
        trade_window = pyautogui.locateOnScreen('images/trade_window.png', confidence=0.8)
        if trade_window:
            print("Trade window found.")
            return True
        else:
            print("Trade window not found.")
            return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# Function to fetch prices
def fetch_prices():
    url = "https://originvalues.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    items = []

    logging.debug(soup.prettify())

    item_containers = soup.find_all(class_="grid-item")
    for container in item_containers:
        try:
            name = container.find("h2").get_text(strip=True)
            price = container.find("span", class_="hc-value").get_text(strip=True)
            image_url = container.find("img")["src"]
            items.append((name, price, image_url))
        except Exception as e:
            logging.error(f"Error parsing item container: {str(e)}")
            logging.error(traceback.format_exc())

    logging.info(f"Fetched items: {items}")
    return items

# GUI Implementation
class BotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TradeBot")
        self.running = False
        self.stop_event = threading.Event()

        try:
            self.bg_image = Image.open("Background.png")
            self.bg_image = self.bg_image.resize((400, 400), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        except Exception as e:
            logging.error(f"Failed to load background image: {str(e)}")
            print(f"Failed to load background image: {str(e)}")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.main_frame = tk.Frame(self.notebook, width=400, height=400)
        self.prices_frame = tk.Frame(self.notebook, width=400, height=400)
        self.main_frame.pack(fill='both', expand=True)
        self.prices_frame.pack(fill='both', expand=True)

        self.notebook.add(self.main_frame, text='Main')
        self.notebook.add(self.prices_frame, text='Prices')

        self.canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)

        self.item_label = tk.Label(self.main_frame, text="Item to Trade:")
        self.item_label.place(x=10, y=10)
        self.item_var = tk.StringVar()
        self.item_dropdown = ttk.Combobox(self.main_frame, textvariable=self.item_var)
        self.item_dropdown['values'] = list(item_templates.keys())
        self.item_dropdown.place(x=120, y=10)

        self.item_count_label = tk.Label(self.main_frame, text="Count:")
        self.item_count_label.place(x=10, y=40)
        self.item_count_entry = tk.Entry(self.main_frame, width=10)
        self.item_count_entry.place(x=120, y=40)
        self.item_count_entry.insert(0, "Max. 99")
        self.item_count_entry.bind("<FocusIn>", self.clear_placeholder)
        self.item_count_entry.bind("<FocusOut>", self.set_placeholder)

        self.want_item_label = tk.Label(self.main_frame, text="Desired Item:")
        self.want_item_label.place(x=10, y=70)
        self.want_item_var = tk.StringVar()
        self.want_item_dropdown = ttk.Combobox(self.main_frame, textvariable=self.want_item_var)
        self.want_item_dropdown['values'] = list(item_templates.keys())
        self.want_item_dropdown.place(x=120, y=70)

        self.want_item_count_label = tk.Label(self.main_frame, text="Count:")
        self.want_item_count_label.place(x=10, y=100)
        self.want_item_count_entry = tk.Entry(self.main_frame, width=10)
        self.want_item_count_entry.place(x=120, y=100)
        self.want_item_count_entry.insert(0, "Max. 99")
        self.want_item_count_entry.bind("<FocusIn>", self.clear_placeholder)
        self.want_item_count_entry.bind("<FocusOut>", self.set_placeholder)

        self.start_button = tk.Button(self.main_frame, text="Start", command=self.toggle_bot)
        self.start_button.place(x=10, y=130)

        self.trade_count_label = tk.Label(self.main_frame, text="Trades Completed: 0")
        self.trade_count_label.place(x=10, y=160)
        self.trades_completed = 0

        self.strategy_label = tk.Label(self.main_frame, text="Strategy Suggestions (Fetch):")
        self.strategy_label.place(x=220, y=10)
        self.strategy_text = tk.Text(self.main_frame, wrap=tk.WORD, width=20, height=10)
        self.strategy_text.place(x=220, y=30)
        self.fetch_strategy_suggestions()

        self.prices_text = tk.Text(self.prices_frame, wrap=tk.WORD)
        self.prices_text.pack(fill='both', expand=True)
        self.fetch_prices_button = tk.Button(self.prices_frame, text="Fetch Prices", command=self.update_prices)
        self.fetch_prices_button.pack()

    def toggle_bot(self):
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop")
            item = self.item_var.get()
            item_count = int(self.item_count_entry.get().replace("Max. ", "").strip())
            want_item = self.want_item_var.get()
            want_item_count = int(self.want_item_count_entry.get().replace("Max. ", "").strip())
            self.stop_event.clear()
            time.sleep(5)
            self.bot_thread = threading.Thread(target=perform_trade,
                                               args=(item, item_count, want_item, want_item_count, self.stop_event))
            self.bot_thread.start()
        else:
            self.running = False
            self.start_button.config(text="Start")
            self.stop_event.set()

    def update_trade_count(self):
        self.trades_completed += 1
        self.trade_count_label.config(text=f"Trades Completed: {self.trades_completed}")

    def update_prices(self):
        self.prices_text.delete(1.0, tk.END)
        items = fetch_prices()
        if items:
            for name, price, image_url in items:
                self.prices_text.insert(tk.END, f"{name}: {price}\n")
        else:
            self.prices_text.insert(tk.END, "No prices found.\n")

    def clear_placeholder(self, event):
        if "Max." in event.widget.get():
            event.widget.delete(0, tk.END)

    def set_placeholder(self, event):
        if event.widget.get() == "":
            event.widget.insert(0, "Max. 99")

    def fetch_strategy_suggestions(self):
        try:
            items = fetch_prices()
            if items:
                suggestions = []
                for name, price, image_url in items:
                    if "Dino Egg" in name:
                        suggestions.append(f"SELL 1 DINO EGG -> BUY 3 PETALS -> SELL 3 PETALS -> BUY 1 MAJESTIC CHAIR -> SELL 1 MAJESTIC CHAIR -> BUY 3 DINO EGGS")
                self.strategy_text.delete(1.0, tk.END)
                for suggestion in suggestions:
                    self.strategy_text.insert(tk.END, f"{suggestion}\n")
            else:
                self.strategy_text.insert(tk.END, "No strategies found.\n")
        except Exception as e:
            logging.error(f"Failed to fetch strategy suggestions: {str(e)}")
            self.strategy_text.insert(tk.END, "Failed to fetch strategies.\n")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = BotApp(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"Script terminated unexpectedly: {str(e)}")
        logging.critical(traceback.format_exc())
        print(f"Script terminated unexpectedly: {str(e)}")
        sys.exit(1)
