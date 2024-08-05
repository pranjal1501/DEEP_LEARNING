import cv2

def get_solid_color(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert image to RGB (OpenCV loads images in BGR format by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the dimensions of the image
    height, width, _ = image_rgb.shape
    
    # Calculate the average color
    average_color = image_rgb.mean(axis=0).mean(axis=0)
    
    # Convert average color to integer
    average_color = average_color.astype(int)
    
    return tuple(average_color)

# Path to your cropped image
image_path = 'E:/shortest_path/images/road_color.png'  # replace with your image path

# Get the solid color
solid_color_rgb = get_solid_color(image_path)

# Print RGB values
print(f"The RGB color of the image is: {solid_color_rgb}")
