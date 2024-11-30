import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    image = cv2.resize(image, (400, 400))  # Resize to 400x400 for simplicity
    
    # Apply Gaussian Blur to reduce noise
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert to Lab color space for better clustering
    image_lab = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2Lab)
    
    return image, image_lab

def kmeans_clustering(image, k):
    # Reshape the image for clustering
    pixel_values = image.reshape((-1, 3))  # Flatten image to 2D array (pixels x channels)
    pixel_values = np.float32(pixel_values)  # Convert to float for K-means
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixel_values)  # Cluster assignments
    centers = np.uint8(kmeans.cluster_centers_)  # Cluster centers (color values)
    
    # Reconstruct the segmented image
    segmented_image = centers[labels].reshape(image.shape)
    
    return segmented_image

def display_segmented_images(original, segmented_images, k_values):
    plt.figure(figsize=(15, 8))
    plt.subplot(1, len(segmented_images) + 1, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")
    
    for i, (segmented_image, k) in enumerate(zip(segmented_images, k_values)):
        plt.subplot(1, len(segmented_images) + 1, i + 2)
        plt.imshow(segmented_image)
        plt.title(f"K={k}")
        plt.axis("off")
    
    plt.show()

def main():
    image_path = "coral_reef_image.jpg"  # Replace with your image path
    
    # Preprocess the image
    original, preprocessed = preprocess_image(image_path)
    
    if preprocessed is None:
        return
    
    # Perform K-means clustering with different k values
    k_values = [2, 5, 8]
    segmented_images = [kmeans_clustering(preprocessed, k) for k in k_values]
    
    # Display the results
    display_segmented_images(original, segmented_images, k_values)

if __name__ == "__main__":
    main()
