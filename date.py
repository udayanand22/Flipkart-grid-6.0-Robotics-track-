import cv2
import time
import re
from datetime import datetime
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import config  # Your Azure config
import threading

# Set up Azure Computer Vision client using config.py
endpoint = config.VISION_ENDPOINT
key = config.VISION_KEY

client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

# Global variable to store OCR results
ocr_result = None

# Capture and process frames in real-time
def process_frame(frame):
    global ocr_result
    # Save the frame to disk as a temporary file
    image_path = "temp_frame.png"
    cv2.imwrite(image_path, frame)
    
    # Process image using OCR
    with open(image_path, "rb") as image_file:
        read_operation = client.read_in_stream(image_file, raw=True)
    
    operation_location = read_operation.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    if result.status == 'succeeded':
        extracted_texts = []
        for read_result in result.analyze_result.read_results:
            for line in read_result.lines:
                extracted_text = line.text.strip()
                extracted_texts.append(extracted_text)

        # Extract and classify dates
        dates = extract_dates(extracted_texts)
        if dates:
            manufacturing_date, expiry_date = classify_dates(extracted_texts, dates)
            expiry_status = check_expiry(expiry_date)
            ocr_result = (manufacturing_date, expiry_date, expiry_status)
        else:
            ocr_result = None
    else:
        ocr_result = None

# Function to extract dates using regex
def extract_dates(extracted_texts):
    dates = []
    date_patterns = [
        r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
        r"\bDATE OF MFG\.\s*(\d{1,2}/\d{1,2}/\d{2,4})\b",
        r"\bUSE BY:\s*(\d{1,2}/\d{1,2}/\d{2,4})\b"
    ]

    for text in extracted_texts:
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match if isinstance(match, str) else match[0]
                dates.append(date_str)
    return dates

# Classify dates into manufacturing and expiry
def classify_dates(extracted_texts, dates):
    manufacturing_date, expiry_date = None, None

    for i, text in enumerate(extracted_texts):
        for date_str in dates:
            if "MFG" in text.upper() or "DATE OF MFG" in text.upper():
                manufacturing_date = parse_date(date_str)
            elif "USE BY" in text.upper() or "EXP" in text.upper():
                expiry_date = parse_date(date_str)

    if not manufacturing_date or not expiry_date:
        parsed_dates = [parse_date(date_str) for date_str in dates if parse_date(date_str)]
        parsed_dates.sort()
        if len(parsed_dates) >= 2:
            return parsed_dates[0], parsed_dates[-1]

    return manufacturing_date, expiry_date

# Parse date from string
def parse_date(date_str):
    date_formats = ["%d.%m.%y", "%d.%m.%Y", "%b %y", "%b %Y", "%m/%Y", "%Y-%m-%d", "%d/%m/%y", "%d/%m/%Y"]
    
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str.strip(), date_format)
        except ValueError:
            continue
    return None

# Check expiry status
def check_expiry(expiry_date):
    if expiry_date:
        today = datetime.today()
        return "Expired" if expiry_date < today else "Not Expired"
    return "Unknown"

# Display real-time OCR results on video feed
def display_results(frame, ocr_result):
    if ocr_result:
        manufacturing_date, expiry_date, expiry_status = ocr_result
        manufacturing_str = manufacturing_date.strftime("%d.%m.%Y") if manufacturing_date else "Unknown"
        expiry_str = expiry_date.strftime("%d.%m.%Y") if expiry_date else "Unknown"
        status_str = expiry_status if expiry_status else "Unknown"

        # Overlay text on the frame
        cv2.putText(frame, f"Manufacturing Date: {manufacturing_str}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Expiry Date: {expiry_str}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {status_str}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Main function to capture frames and process OCR in real-time
def main():
    global ocr_result
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Use a separate thread to process frames for OCR
    def process_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame)  # Process the current frame for OCR

    threading.Thread(target=process_frames, daemon=True).start()

    # Continuously display the camera feed and overlay OCR results
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display OCR results on the frame
        display_results(frame, ocr_result)

        cv2.imshow("Real-time OCR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
