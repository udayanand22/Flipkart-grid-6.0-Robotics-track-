import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# Replace with your actual details
prediction_key = config.VISION_KEY
endpoint = config.VISION_ENDPOINT
project_id = "dfa70bb2-5e65-4bde-b1f9-f495daa65e85"
iteration_name = "Iteration3"  # Make sure this is the correct name of your published iteration

# Function to predict using an image file
def predict_image_file(image_path):
    url = f"{endpoint}/customvision/v3.0/Prediction/{project_id}/detect/iterations/{iteration_name}/image"
    
    headers = {
        "Prediction-Key": prediction_key,
        "Content-Type": "application/octet-stream"
    }
    
    # Read the image file
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None

    try:
        response = requests.post(url, headers=headers, data=image_data)
        response.raise_for_status()  # Raise an error for bad responses
        predictions = response.json()
        return predictions
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        print(f"Response content: {response.text}")
        return None

# Function to display predictions using Matplotlib
def display_predictions(image_path, predictions, product_counts):
    # Load the image
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    # Create a patch for each prediction
    for prediction in predictions['predictions']:
        if prediction['probability'] > 0.5:  # Filter based on probability threshold
            box = prediction['boundingBox']
            rect = patches.Rectangle((box['left'] * image.shape[1], box['top'] * image.shape[0]), 
                                     box['width'] * image.shape[1], box['height'] * image.shape[0],
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Position the text label
            label_x = box['left'] * image.shape[1]
            label_y = (box['top'] * image.shape[0]) - 10  # Move the label slightly above the box

            # Add a background for better visibility
            ax.text(label_x, label_y, f"{prediction['tagName']} ({prediction['probability']:.2f})",
                    fontsize=12, color='red', weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Display the product counts vertically on the right-hand side of the image
    count_text_x = image.shape[1] + 10  # Offset the text from the image
    count_text_y = 20  # Start at the top and increment downward
    for product, count in product_counts.items():
        ax.text(count_text_x, count_text_y, f"{product}: {count}", fontsize=12, color='blue', weight='bold')
        count_text_y += 20  # Increment the y-position for each product count

    ax.axis('off')  # Hide axes
    plt.title('Predictions and Product Counts')
    plt.show()

# Function to count occurrences of each product
def count_products(predictions):
    product_counts = Counter()
    for prediction in predictions['predictions']:
        if prediction['probability'] > 0.5:  # Filter based on probability threshold
            product_counts[prediction['tagName']] += 1
    return product_counts

# Example usage with an image file
image_path = "counting.png"  # Replace with the path to your actual image file
predictions = predict_image_file(image_path)

if predictions:
    print(predictions)
    
    product_counts = count_products(predictions)
    print("Product counts:", product_counts)
    
    display_predictions(image_path, predictions, product_counts)
else:
    print("No predictions returned.")
