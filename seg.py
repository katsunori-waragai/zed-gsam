import numpy as np

def points_by_segmentation(points: np.ndarray, segmentation_image: np.ndarray):
    """
    segmentationは結果を元に、対応する点群の範囲を返す。
    points: height, width, channel の構成
    pointsは添字の順番がheight, width, channelの順番である。
    chanelには、X, Y, Z, colorが含まれている。
    segmentation_imageは、height, width のデータ
    segmentationの添字はheight, width の順番である。
    セグメンテーションの分類はuint8 の整数で分類済みである。

    戻り値は、各セグメンテーションに対応するpointsのsubsetのリストを返す。
    """
    
    # Get unique segmentation labels
    unique_labels = np.unique(segmentation_image)
    
    # Initialize a list to hold points for each segmentation label
    segmented_points = []
    
    # Iterate through unique labels and collect corresponding points
    for label in unique_labels:
        mask = segmentation_image == label
        labeled_points = points[mask]
        segmented_points.append(labeled_points)
    
    return segmented_points

# Example usage
if __name__ == "__main__":
    # Example points array (height, width, channels)
    points = np.random.rand(4, 4, 4)  # Random points for demonstration
    # Example segmentation image (height, width)
    segmentation_image = np.random.randint(0, 3, (4, 4))  # Random segmentation labels (0, 1, 2)
    
    segmented_points_list = points_by_segmentation(points, segmentation_image)
    for idx, segment in enumerate(segmented_points_list):
        print(f"Segment {idx} points:\n", segment)

