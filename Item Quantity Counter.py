import cv2
import numpy as np
import pyautogui
import time
import logging
import json
import os
from datetime import datetime

# Initialize logging for error tracking
logging.basicConfig(filename='item_quantity_counter_errors.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Directory to save screenshots
output_dir = r"images"
os.makedirs(output_dir, exist_ok=True)

# Path to the blank reference image
blank_image_path = r"Blank.png"

# Path to the JSON file for sharing item counts
item_counts_file = r"item_counts.json"

# Coordinates for the trade slots
your_slot_coords = (1640, 497, 21, 20)
their_slot_coords = (1306, 498, 21, 20)

# Function to save item counts to a JSON file
def save_item_counts(your_item_count, their_item_count):
    data = {
        "your_item_count": your_item_count,
        "their_item_count": their_item_count
    }
    with open(item_counts_file, 'w') as f:
        json.dump(data, f)

# Function to load templates
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

# Function to preprocess an image
def preprocess_image(image):
    if len(image.shape) == 2:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Function to match a template
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

# Function to check if the slot image is blank
def is_blank_image(image, blank_image_path):
    gray_image = preprocess_image(image)
    blank_image = preprocess_image(cv2.imread(blank_image_path))

    # Resize the blank image to match the slot image size if needed
    if gray_image.shape != blank_image.shape:
        blank_image = cv2.resize(blank_image, (gray_image.shape[1], gray_image.shape[0]))

    # Check if the image matches the blank reference image
    is_blank = np.array_equal(gray_image, blank_image)
    logging.info(f"Is blank image: {is_blank}")
    return is_blank

# Function to read a number from a trade slot
def read_number_from_trade_slot(screen, slot_coords, templates, label):
    x, y, w, h = slot_coords
    slot_region = screen[y:y + h, x:x + w]
    gray_slot = preprocess_image(slot_region)
    if is_blank_image(slot_region, blank_image_path):
        return 0
    matched_number = match_template(gray_slot, templates)
    if matched_number is None or matched_number == 'Empty':
        return 0
    return int(matched_number.split('_')[-1])

# Main function
def main():
    try:
        templates_me, templates_other = load_templates()
        while True:
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            your_item_count = read_number_from_trade_slot(screenshot, your_slot_coords, templates_me, 'Your_Item')
            print(f"MY item COUNT: {your_item_count}")

            their_item_count = read_number_from_trade_slot(screenshot, their_slot_coords, templates_other, 'Their_Item')
            print(f"OTHER players item COUNT: {their_item_count}")

            save_item_counts(your_item_count, their_item_count)

            time.sleep(1)  # Adjust the frequency as needed

    except Exception as e:
        logging.error(f"Bot encountered an error: {str(e)}")
        print(f"Bot encountered an error: {str(e)}")

if __name__ == "__main__":
    main()
