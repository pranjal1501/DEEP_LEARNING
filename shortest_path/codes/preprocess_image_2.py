import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy.spatial import distance

# Function to check if the color is similar to the target color within the given tolerance
def is_similar_color(color, target_color, tolerance):
    return np.all(np.abs(color - target_color) <= tolerance)

# Function to convert the target color to green in the image
def convert_image(image_path, target_rgb, tolerance=(30, 30, 30), green_rgb=(0, 255, 0)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.apply_along_axis(is_similar_color, -1, image_rgb, target_rgb, tolerance)
    result_image = np.zeros_like(image_rgb)
    result_image[mask] = green_rgb
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    binary_image_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    result_image[~mask] = binary_image_rgb[~mask]
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    return result_image_bgr

# Function to check if the pixel is green
def is_green(pixel, green_rgb=(0, 255, 0)):
    return np.all(pixel == green_rgb)

# Function to process the converted image by blurring green pixels
def process_converted_image(image_path, green_rgb=(0, 255, 0)):
    image = cv2.imread(image_path)
    mask = np.apply_along_axis(is_green, -1, image, green_rgb)
    mask_3d = np.stack([mask] * 3, axis=-1).astype(np.uint8) * 255
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    result_image = np.where(mask_3d, blurred_image, image)
    return result_image

# Function to convert colors within the target range to green
def convert_colors(image_path, target_color=(0, 255, 0), tolerance=(30, 30, 30)):
    image = cv2.imread(image_path)
    lower_color = np.array([max(0, target_color[0] - tolerance[0]),
                            max(0, target_color[1] - tolerance[1]),
                            max(0, target_color[2] - tolerance[2])])
    upper_color = np.array([min(255, target_color[0] + tolerance[0]),
                            min(255, target_color[1] + tolerance[1]),
                            min(255, target_color[2] + tolerance[2])])
    mask = cv2.inRange(image, lower_color, upper_color)
    result_image = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    result_image[mask > 0] = target_color
    return result_image

# Function to convert the target color to black in the image
def convert_color_to_black(image_path, target_color, tolerance):
    image = cv2.imread(image_path)
    lower_bound = np.array([target_color[0] - tolerance[0], target_color[1] - tolerance[1], target_color[2] - tolerance[2]])
    upper_bound = np.array([target_color[0] + tolerance[0], target_color[1] + tolerance[1], target_color[2] + tolerance[2]])
    mask = cv2.inRange(image, lower_bound, upper_bound)
    image[mask != 0] = [0, 0, 0]
    return image

# Function to extend black regions in the binary image
def extend_black_regions(binary_image, extension):
    binary_processed = binary_image.copy()
    height, width = binary_image.shape
    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 0:
                for dy in range(-extension, extension + 1):
                    for dx in range(-extension, extension + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            binary_processed[ny, nx] = 0
    return binary_processed

# Function to perform skeletonization (thinning) on the image
def skeletonize_image(thick_image_path):
    thick_image = cv2.imread(thick_image_path, cv2.IMREAD_GRAYSCALE)
    _, thick_binary = cv2.threshold(thick_image, 127, 255, cv2.THRESH_BINARY)
    thick_binary = cv2.bitwise_not(thick_binary)
    bool_image = thick_binary.astype(bool)
    skeleton = skeletonize(bool_image)
    thin_image = (skeleton * 255).astype(np.uint8)
    thin_image = cv2.bitwise_not(thin_image)
    return thin_image

# Function to find intersections by checking the 3x3 neighborhood
def find_intersections(skeleton_image):
    intersections = []
    for y in range(1, skeleton_image.shape[0] - 1):
        for x in range(1, skeleton_image.shape[1] - 1):
            if skeleton_image[y, x] == 255:
                neighborhood = skeleton_image[y-1:y+2, x-1:x+2]
                num_connections = np.sum(neighborhood == 255) - 1
                if num_connections >= 3:
                    intersections.append((x, y))
    return intersections

# Function to cluster close intersections
def cluster_intersections(intersections, distance_threshold=20):
    clustered_intersections = []
    while intersections:
        base_point = intersections.pop(0)
        cluster = [base_point]
        for point in intersections[:]:
            if distance.euclidean(base_point, point) < distance_threshold:
                cluster.append(point)
                intersections.remove(point)
        cluster_mean = np.mean(cluster, axis=0).astype(int)
        clustered_intersections.append(tuple(cluster_mean))
    return clustered_intersections

# Function to overlay marked nodes on an image of choice
def overlay_marked_nodes_on_image(base_image_path, marked_nodes_image_path, output_image_path):
    base_image = cv2.imread(base_image_path)
    marked_nodes_image = cv2.imread(marked_nodes_image_path)
    if base_image.shape[:2] != marked_nodes_image.shape[:2]:
        raise ValueError("Base image and marked nodes image must have the same dimensions")
    overlay_image = cv2.addWeighted(base_image, 0.7, marked_nodes_image, 0.3, 0)
    cv2.imwrite(output_image_path, overlay_image)

# Path to your image
image_path = 'E:/shortest_path/image.jpeg'
target_rgb = (70, 79, 88)
tolerance = (30, 30, 30)

# Convert the image
converted_image = convert_image(image_path, target_rgb, tolerance)
cv2.imwrite('converted_image.jpg', converted_image)

# Process the converted image
processed_image = process_converted_image('converted_image.jpg')
cv2.imwrite('processed_image.jpg', processed_image)

# Convert colors in the processed image
processed_image = convert_colors('processed_image.jpg', target_color=(0, 255, 0), tolerance=(95, 95, 95))
cv2.imwrite('binarized_image.jpg', processed_image)

# Convert green to black in the binarized image
black_image = convert_color_to_black('binarized_image.jpg', target_color=(0, 255, 0), tolerance=(95, 95, 95))
cv2.imwrite('black_image.jpg', black_image)

# Extend black regions
extended_black_image = extend_black_regions(cv2.imread('black_image.jpg', cv2.IMREAD_GRAYSCALE), extension=2)
cv2.imwrite('extended_black_image.jpg', extended_black_image)

# Perform skeletonization to thin the image
thin_image = skeletonize_image('extended_black_image.jpg')
cv2.imwrite('thin_image.jpg', thin_image)

# Load the thinned image
thin_image_path = 'thin_image.jpg'
thin_image = cv2.imread(thin_image_path, cv2.IMREAD_GRAYSCALE)
_, thin_binary = cv2.threshold(thin_image, 127, 255, cv2.THRESH_BINARY)
thin_binary = cv2.bitwise_not(thin_binary)
skeleton = skeletonize(thin_binary // 255)
thin_image_skeleton = (skeleton * 255).astype(np.uint8)

# Find intersections
intersection_points = find_intersections(thin_image_skeleton)

# Cluster close intersections
clustered_intersections = cluster_intersections(intersection_points)

# Load the original thin image in color to draw red nodes
color_image = cv2.cvtColor(thin_image, cv2.COLOR_GRAY2BGR)

# Print the number of detected intersections
print(f"Number of detected intersections: {len(clustered_intersections)}")

# Mark the nodes in red and assign numbers
node_number = 1
for (x, y) in clustered_intersections:
    cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(color_image, str(node_number), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    node_number += 1

# Save the output image
marked_nodes_image_path = 'marked_nodes_image.jpg'
cv2.imwrite(marked_nodes_image_path, color_image)

print(f"Processed image with marked nodes saved as {marked_nodes_image_path}")

# Overlay marked nodes on an image of choice
base_image_path = 'E:/shortest_path/image.jpeg'  # Replace with your base image path
overlay_image_path = 'overlay_marked_nodes_image.jpg'
overlay_marked_nodes_on_image(base_image_path, marked_nodes_image_path, overlay_image_path)

print(f"Overlay image saved as {overlay_image_path}")
print("All processing steps completed successfully!")
