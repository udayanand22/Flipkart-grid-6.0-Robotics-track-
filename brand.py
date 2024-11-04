import os
import cv2
import time
import mysql.connector
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import config  # Import the configuration
import tkinter as tk
from tkinter import messagebox

# Set up Azure Computer Vision client using config.py
endpoint = config.VISION_ENDPOINT
key = config.VISION_KEY

# Initialize the Computer Vision client
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

# Connect to MySQL database
def connect_to_database():
    connection = mysql.connector.connect(
        hhost='localhost',
        user='root',
        password='.......',
        database='Flipkart'
    )
    return connection

# Capture image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    print("Press 'c' to capture an image or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Camera Preview", frame)
        key = cv2.waitKey(1) & 0xFF  # Get the key pressed

        if key == ord('c'):
            image_path = "captured_image.png"
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as {image_path}")
            break
        
        elif key == ord('q'):
            print("Quitting without capturing an image.")
            break

    cap.release()
    cv2.destroyAllWindows()

    return image_path if key == ord('c') else None

# Process image with Azure OCR
def process_image_with_ocr(image_path):
    with open(image_path, "rb") as image_file:
        read_operation = client.read_in_stream(image_file, raw=True)

    operation_location = read_operation.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        print('Waiting for OCR to complete...')
        time.sleep(1)

    if result.status == 'succeeded':
        print("OCR Results:")
        extracted_texts = []
        for read_result in result.analyze_result.read_results:
            for line in read_result.lines:
                extracted_text = line.text.strip().lower()  # Normalize the text
                extracted_texts.append(extracted_text)
                print(f"Text: {line.text}")

        # Query database for matching brands and products
        matched_brands, matched_products = query_database_for_matches(extracted_texts)

        # Show results in a popup dialog
        show_results_popup(matched_brands, matched_products)
    else:
        print("OCR operation failed.")

def show_results_popup(matched_brands, matched_products):
    # Create the main tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    if matched_brands:
        brand_names = ", ".join(matched_brands)
    else:
        brand_names = "No matching brands found."

    if matched_products:
        product_names = ", ".join(matched_products)
    else:
        product_names = "No matching products found."

    # Create the message box
    messagebox.showinfo("OCR Results", f"Matched Brands: {brand_names}\nMatched Products: {product_names}")

    root.destroy()  # Close the tkinter window

def query_database_for_matches(extracted_texts):
    connection = connect_to_database()
    cursor = connection.cursor()

    matched_brands = set()
    matched_products = {}

    # Normalize extracted texts
    normalized_texts = [text.replace("'", "''").strip() for text in extracted_texts]
    print(f"Normalized Texts: {normalized_texts}")  # Debugging line

    # Check for exact matching brands first
    for text in normalized_texts:
        brand_query = f"SELECT brand_name, brand_id FROM brands WHERE brand_name = '{text}'"
        print(f"Executing exact brand query: {brand_query}")  # Debugging line
        cursor.execute(brand_query)
        brand_results = cursor.fetchall()
        print(f"Exact Brand results: {brand_results}")  # Debugging line

        for brand in brand_results:
            matched_brands.add(brand[0])
            brand_id = brand[1]  # Capture the brand_id for product matching

            # Query for products associated with the matched brand
            product_query = f"SELECT product_name FROM products WHERE brand_id = {brand_id}"
            print(f"Executing product query for brand {brand[0]}: {product_query}")  # Debugging line
            cursor.execute(product_query)
            product_results = cursor.fetchall()
            print(f"Product results for brand {brand[0]}: {product_results}")  # Debugging line
            for product in product_results:
                matched_products[product[0]] = matched_products.get(product[0], 0) + 1  # Count matches

    # If no matches found in exact matching, check for broader matching brands
    if not matched_brands:
        for text in normalized_texts:
            brand_query = f"SELECT brand_name, brand_id FROM brands WHERE brand_name LIKE '%{text}%'"
            print(f"Executing broader brand query: {brand_query}")  # Debugging line
            cursor.execute(brand_query)
            brand_results = cursor.fetchall()
            print(f"Broader Brand results: {brand_results}")  # Debugging line

            for brand in brand_results:
                matched_brands.add(brand[0])
                brand_id = brand[1]  # Capture the brand_id for product matching

                # Query for products associated with the matched brand
                product_query = f"SELECT product_name FROM products WHERE brand_id = {brand_id}"
                print(f"Executing product query for brand {brand[0]}: {product_query}")  # Debugging line
                cursor.execute(product_query)
                product_results = cursor.fetchall()
                print(f"Product results for brand {brand[0]}: {product_results}")  # Debugging line
                for product in product_results:
                    matched_products[product[0]] = matched_products.get(product[0], 0) + 1  # Count matches

    # Check for products based on exact match
    if not matched_products:  # Only check if no products have been matched yet
        for text in normalized_texts:
            product_query = f"SELECT p.product_name, b.brand_name FROM products p JOIN brands b ON p.brand_id = b.brand_id WHERE p.product_name = '{text}'"
            print(f"Executing exact product query: {product_query}")  # Debugging line
            cursor.execute(product_query)
            product_results = cursor.fetchall()
            print(f"Exact product results: {product_results}")  # Debugging line
            for product in product_results:
                matched_products[product[0]] = matched_products.get(product[0], 0) + 1  # Count matches
                matched_brands.add(product[1])  # Add corresponding brand

    # If no matches, split the strings and check for matches
    if not matched_products:
        for text in normalized_texts:
            words = text.split()  # Split into individual words for last resort matching
            for word in words:
                # Check for products based on broader match
                product_query = f"SELECT p.product_name, b.brand_name FROM products p JOIN brands b ON p.brand_id = b.brand_id WHERE p.product_name LIKE '%{word}%'"
                print(f"Executing broader product query: {product_query}")  # Debugging line
                cursor.execute(product_query)
                product_results = cursor.fetchall()
                print(f"Broader product results: {product_results}")  # Debugging line
                for product in product_results:
                    matched_products[product[0]] = matched_products.get(product[0], 0) + 1  # Count matches
                    matched_brands.add(product[1])  # Add corresponding brand

    # Select products with the highest match count
    if matched_products:
        # Sort products by match count
        sorted_products = sorted(matched_products.items(), key=lambda item: item[1], reverse=True)
        best_match_product = sorted_products[0][0]  # Get the product with the highest match count
        matched_products = {best_match_product}  # Retain only the best match

    cursor.close()
    connection.close()

    return matched_brands, matched_products

# Main function to capture the image and process it with OCR
def main():
    image_path = capture_image()

    if image_path:
        process_image_with_ocr(image_path)
    else:
        print("No image captured. Exiting.")

# Run the main function
if __name__ == "__main__":
    main()