import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.spatial import distance
import heapq

# Load the thinned image
thin_image_path = "E:/shortest_path/precrocess_steps/thin_image.jpg"
thin_image = cv2.imread(thin_image_path, cv2.IMREAD_GRAYSCALE)

# Load the adjacent nodes CSV
adjacent_nodes_path = "E:/shortest_path/adjacent_nodes.csv"
adjacent_nodes_df = pd.read_csv(adjacent_nodes_path)

# Threshold the image to ensure binary format
_, thin_binary = cv2.threshold(thin_image, 127, 255, cv2.THRESH_BINARY)

# Invert the image to have lines as white on black background
thin_binary = cv2.bitwise_not(thin_binary)

# Perform skeletonization to ensure thin lines
skeleton = skeletonize(thin_binary // 255)

# Convert the skeleton back to uint8 format
thin_image_skeleton = (skeleton * 255).astype(np.uint8)

# Function to find intersections by checking the 3x3 neighborhood
def find_intersections(skeleton_image):
    intersections = []
    for y in range(1, skeleton_image.shape[0] - 1):
        for x in range(1, skeleton_image.shape[1] - 1):
            if skeleton_image[y, x] == 255:
                neighborhood = skeleton_image[y-1:y+2, x-1:x+2]
                num_connections = np.sum(neighborhood == 255) - 1  # Subtract 1 to exclude the center pixel
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

# Find intersections
intersection_points = find_intersections(thin_image_skeleton)

# Cluster close intersections
clustered_intersections = cluster_intersections(intersection_points)

# Create a blank white image
blank_image = np.ones_like(thin_image) * 255

# Load the blank image in color to draw red nodes and lines
color_image = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)

# Mark the nodes in red and assign numbers
node_number = 1
node_positions = {}
for (x, y) in clustered_intersections:
    cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(color_image, str(node_number), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    node_positions[node_number] = (x, y)
    node_number += 1

# Function to draw lines and calculate distances between adjacent nodes
def draw_lines_and_calculate_distances(image, intersections, node_positions):
    distances = []
    for node1, node2 in intersections:
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        dist = distance.euclidean((x1, y1), (x2, y2))
        distances.append(dist)
    return distances

# Draw lines and calculate distances
distances = draw_lines_and_calculate_distances(color_image, adjacent_nodes_df[['Node1', 'Node2']].values, node_positions)

# Add distances to the DataFrame
adjacent_nodes_df['Distance'] = distances

# Save the updated DataFrame to a new CSV file
updated_csv_path = "adjacent_nodes_with_distances.csv"
adjacent_nodes_df.to_csv(updated_csv_path, index=False)

# Dijkstra's algorithm to find the shortest path
def dijkstra(graph, start, end):
    queue, seen, mins = [(0, start, [])], set(), {start: 0}
    while queue:
        (cost, v1, path) = heapq.heappop(queue)
        if v1 not in seen:
            seen.add(v1)
            path = path + [v1]
            if v1 == end:
                return (cost, path)
            for c, v2 in graph.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heapq.heappush(queue, (next, v2, path))
    return float("inf"), []

# Build graph from the DataFrame
graph = {}
for idx, row in adjacent_nodes_df.iterrows():
    node1, node2, dist = int(row['Node1']), int(row['Node2']), row['Distance']
    if node1 not in graph:
        graph[node1] = []
    if node2 not in graph:
        graph[node2] = []
    graph[node1].append((dist, node2))
    graph[node2].append((dist, node1))

# Get user input for start and end nodes
start_node = int(input("Enter the start node: "))
end_node = int(input("Enter the end node: "))

# Find the shortest path using Dijkstra's algorithm
shortest_distance, shortest_path = dijkstra(graph, start_node, end_node)

# Highlight the shortest path on the image
for i in range(len(shortest_path) - 1):
    x1, y1 = node_positions[shortest_path[i]]
    x2, y2 = node_positions[shortest_path[i + 1]]
    cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the output image with the shortest path highlighted
output_image_with_path_path = "marked_nodes_with_shortest_path_image.jpg"
cv2.imwrite(output_image_with_path_path, color_image)

# Load the background image
background_image_path = "E:/shortest_path/image.jpeg"  # Replace with the path to your background image
background_image = cv2.imread(background_image_path)

# Resize the background image to match the processed image size
background_image = cv2.resize(background_image, (color_image.shape[1], color_image.shape[0]))

# Overlay the processed image onto the background image
alpha = 0.5  # Adjust the transparency of the overlay
overlay_image = cv2.addWeighted(background_image, 1 - alpha, color_image, alpha, 0)

# Save the final overlaid image
final_output_image_path = "final_overlay_image.jpg"
cv2.imwrite(final_output_image_path, overlay_image)

# Print the shortest distance and the path
print(f"The shortest distance from node {start_node} to node {end_node} is {shortest_distance}.")
print(f"The path is: {shortest_path}")

print(f"Processed image with the shortest path highlighted saved as {output_image_with_path_path}")
print(f"Final overlay image saved as {final_output_image_path}")
