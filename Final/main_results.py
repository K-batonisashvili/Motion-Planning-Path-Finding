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

        :param skeleton: The skeleton of an image to use for tracing.
        :param gif_path: Saving the image in a GIF format.
        :return: Returns the path parameter which describes where the robot must move in the physical world.
                 The return param is a part of the software pipeline which converts digital coordinates to
                 the physical world.
        """
        # Create a binary map of the uploaded
        skeleton_map = skeleton > 0  # True where skeleton pixels exist
        traced_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

        # List to store frames for the GIF
        frames = []
        frame_counter = 0

        # Find all skeleton points (non-zero pixels)
        skeleton_points = np.argwhere(skeleton_map)
        visited = set()

        # Start tracing from the first skeleton point. This is typically top left as that is how images are processed
        current_point = tuple(skeleton_points[0])  # (row, col)
        path = [current_point]

        while skeleton_points.size > 0:
            visited.add(current_point)
            skeleton_map[current_point] = False  # Mark the point as traced

            # Draw the current point
            traced_image = cv2.circle(traced_image, (current_point[1], current_point[0]), 1, (0, 0, 255), -1)

            # Save every 20th frame for the GIF. Frame buffer can be adjusted for more FPS.
            if frame_counter == 0:
                gif_frame = cv2.cvtColor(traced_image, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(gif_frame))
            elif frame_counter % self.frame_buffer == 0:
                gif_frame = cv2.cvtColor(traced_image, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(gif_frame))

            # Display the tracing progress in real-time. This can be commented out if not necessary.
            cv2.imshow("Tracing Skeleton", traced_image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit early
                break

            # Find neighbors in 8-connected neighborhood
            neighbors = self.get_neighbors(current_point, skeleton_map)

            if neighbors:
                # Move to the nearest unvisited_points neighbor. This is the DFS algorithm
                next_point = neighbors[0]
                color = (0, 0, 255)  # Red for direct neighbors
            else:
                # Command checks if there are any unvisited points left.
                unvisited_points = np.argwhere(skeleton_map)
                if unvisited_points.size == 0:
                    break  # All pixels are traced

                # If there are no direct neighbors, the BFS algorithm kicks in which determins the
                # closest next point through Euclidean distance (This is a jump).
                BFS_distance = np.linalg.norm(unvisited_points - np.array(current_point), axis=1)
                nearest_idx = np.argmin(BFS_distance)
                next_point = tuple(unvisited_points[nearest_idx])
                color = (0, 255, 0)  # Green for jumps

            # Draw a line to the next point
            traced_image = cv2.line(
                traced_image, (current_point[1], current_point[0]), (next_point[1], next_point[0]), color, 1
            )

            # Move to the next point
            current_point = next_point
            frame_counter += 1
            path.append(current_point)

        # GIF saving code. Comment this out if not needed.
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # Set duration per frame (in ms, adjust for FPS)
            loop=0  # Infinite loop
        )

        cv2.destroyAllWindows()
        return path

    def run_simulation(self):
        """
        The actual run command passed to the simulation so that an image gets converted
        to a skeletonized form, and then all segments get traced.
        """
        skeleton = self.preprocess_image()  # Get the skeletonized image
        print("Starting tracing...")
        path = self.trace_skeleton(skeleton)
        print("Tracing completed. Path length:", len(path))

    def calculate_path_length(path):
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
