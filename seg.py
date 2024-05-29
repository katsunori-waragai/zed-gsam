from typing import List

import numpy as np

from depth_and_gsam import points_by_segmentation

# Example usage
if __name__ == "__main__":
    # Example points array (height, width, channels)
    points = np.random.rand(4, 4, 4)  # Random points for demonstration
    # Example segmentation image (height, width)
    segmentation_image = np.random.randint(0, 3, (4, 4), dtype=np.uint8)  # Random segmentation labels (0, 1, 2)
    
    segmented_points_list = points_by_segmentation(points, segmentation_image)
    for idx, segment in enumerate(segmented_points_list):
        print(f"Segment {idx} points:\n", segment)

