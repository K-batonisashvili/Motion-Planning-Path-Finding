import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image


class ContinuousLineTracer:
    """
    Main class for creation of skeletonized image, hybrid DFS-BFS planning, and returning a usable path array
    which contains all movable locations for the tracing of the entire configuration space.
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Image file not found or unable to load: {image_path}")
        self.traced_image = None
        self.frame_buffer = 20  # Frame buffer can be adjusted to change how many FPS are saved for the GIF

    def get_neighbors(self, point, skeleton_map):
        """
        :param point: Current point of where we are on the map.
        :param skeleton_map: Skeleton map that was extracted from the image.
        :return: Nearby neighbors for Path Planning.
        """

        row, col = point
        neighbors = []

        # Check all 8-connected neighbors, starting from top left to bottom right
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < skeleton_map.shape[0] and 0 <= c < skeleton_map.shape[1] and skeleton_map[r, c]:
                neighbors.append((r, c))

        return neighbors

    def preprocess_image(self):
        """
        :return: Returns the skeleton of an image.
        """
        # Converts pixels with intensity 128 to white space and those below into black space
        _, binary = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to close pixel gaps and maintain continuity
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Convert to boolean for skeletonization
        binary_bool = closed.astype(bool)

        # Skeletonize the image
        skeleton = skeletonize(binary_bool)

        # Convert back to uint8 format for display and further processing
        skeleton = (skeleton * 255).astype(np.uint8)

        # cv2.imshow("Skeletonized Image", skeleton)
        return skeleton

    def trace_skeleton(self, skeleton, gif_path="tracing_simulation.gif"):
        """
        Traces the skeleton of an image using a hybrid DFS-BFS algorithm, measures path lengths,
        and draws the path on the image.

        :param skeleton: The skeleton of an image to use for tracing.
        :param gif_path: Path to save the GIF of the tracing process.
        :return: Skeleton path, full coverage path, skeleton length, full coverage length
        """
        skeleton_map = skeleton > 0  # True where skeleton pixels exist
        traced_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

        skeleton_path = []  # Path from DFS-only tracing
        full_coverage_path = []  # Path including BFS jumps

        skeleton_points = np.argwhere(skeleton_map)
        visited = set()

        current_point = tuple(skeleton_points[0])  # Starting point
        skeleton_path.append(current_point)
        full_coverage_path.append(current_point)

        # List to store frames for the GIF
        frames = []
        frame_counter = 0

        while skeleton_points.size > 0:
            visited.add(current_point)
            skeleton_map[current_point] = False  # Mark the point as traced

            neighbors = self.get_neighbors(current_point, skeleton_map)

            if neighbors:
                # Move to the nearest neighbor (DFS)
                next_point = neighbors[0]
            else:
                # No neighbors left, perform BFS to find the nearest unvisited point
                unvisited_points = np.argwhere(skeleton_map)
                if unvisited_points.size == 0:
                    break  # All points are traced

                BFS_distance = np.linalg.norm(unvisited_points - np.array(current_point), axis=1)
                nearest_idx = np.argmin(BFS_distance)
                next_point = tuple(unvisited_points[nearest_idx])

            # Draw the current point
            if neighbors:
                traced_image = cv2.circle(traced_image, (current_point[1], current_point[0]), 1, (0, 0, 255),
                                          -1)  # Red for DFS
            else:
                traced_image = cv2.circle(traced_image, (current_point[1], current_point[0]), 1, (0, 255, 0),
                                          -1)  # Green for BFS

            # Add the path to the respective lists
            skeleton_path.append(next_point) if neighbors else None
            full_coverage_path.append(next_point)

            # Draw a line to the next point
            traced_image = cv2.line(
                traced_image, (current_point[1], current_point[0]), (next_point[1], next_point[0]),
                (0, 0, 255) if neighbors else (0, 255, 0), 1
            )

            # Save every 20th frame for the GIF
            if frame_counter == 0:
                gif_frame = cv2.cvtColor(traced_image, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(gif_frame))
            elif frame_counter % self.frame_buffer == 0:
                gif_frame = cv2.cvtColor(traced_image, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(gif_frame))

            # Update current point
            current_point = next_point
            frame_counter += 1

        # GIF saving code. Comment this out if not needed.
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # Set duration per frame (in ms, adjust for FPS)
            loop=0  # Infinite loop
        )

        cv2.destroyAllWindows()

        # Calculate path lengths
        skeleton_length = self.calculate_path_length(skeleton_path)
        full_coverage_length = self.calculate_path_length(full_coverage_path)

        return skeleton_path, full_coverage_path, skeleton_length, full_coverage_length

    def run_simulation(self):
        """
        Simulates the skeleton tracing and measures the path lengths.
        """
        skeleton = self.preprocess_image()  # Get the skeletonized image
        print("Starting tracing...")

        skeleton_path, full_coverage_path, skeleton_length, full_coverage_length = self.trace_skeleton(skeleton)

        print("Tracing completed.")
        print(f"Skeleton Path Length: {skeleton_length:.2f} pixels")
        print(f"Full Coverage Path Length: {full_coverage_length:.2f} pixels")
        print(f"Additional Length Due to Jumps: {full_coverage_length - skeleton_length:.2f} pixels")

    def calculate_path_length(self, path):
        """
        Calculates the total length of a given path using Euclidean distance.
        :param path: List of (row, col) coordinates.
        :return: Total path length.
        """
        total_length = 0.0
        for i in range(1, len(path)):
            p1, p2 = path[i - 1], path[i]
            total_length += np.linalg.norm(np.array(p2) - np.array(p1))
        return total_length


if __name__ == "__main__":
    """Initialization of the code"""
    # ENSURE THAT THE IMAGE IS IN THE SAME DIRECTORY AS THIS CODE
    image_path = "Soccer.jpg"  # Replace this with desired configuration space for tracing
    line_tracer = ContinuousLineTracer(image_path)
    line_tracer.run_simulation()
