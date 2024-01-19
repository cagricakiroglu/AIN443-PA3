import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import time


try:
    SIFT = cv2.SIFT_create()
except AttributeError:
    # If you don't have SIFT, sometimes it might be under a different namespace
    SIFT = cv2.xfeatures2d.SIFT_create()

plt.style.use("ggplot")
DATASET_NAME = "src/dataset"

BRIGHTNESS_FACTOR = 2.4
NEAREST_NEIGHBOR_NUM = 2
RANSAC_THRESH = 3.0
INLIER_THRESH = 0.7 
RANSAC_ITERATION =1000
subsets = {}
index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks=100)
# Matchers
brute_force_matcher = cv2.BFMatcher()
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)



def get_subset_names(dataset_path):
    """
    Populate a dictionary with subset names and their corresponding image filenames.

    This function iterates through the directories in the dataset path and stores
    the names of each subset along with a sorted list of image filenames in each subset.

    :param dataset_path: Path to the dataset directory.
    :return: A dictionary with subset names as keys and lists of image filenames as values.
    """
    subsets = {}
    try:
        for directory in sorted(os.listdir(dataset_path)):
            dir_path = os.path.join(dataset_path, directory)
            if os.path.isdir(dir_path):
                files = sorted(file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file)))
                subsets[directory] = files
    except FileNotFoundError:
        print(f"Error: The directory '{dataset_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return subsets

def main():
    
    subsets = get_subset_names(DATASET_NAME)
    
    if subsets:
        for subset_name, subset_image_names in subsets.items():
            process_subset(DATASET_NAME, subset_name, subset_image_names)

    print( sum(carmel) / len(carmel) ) 
    print( sum(fishbowl) / len(fishbowl) ) 
    print( sum(goldengate) / len(goldengate) ) 
    print( sum(hotel) / len(hotel) ) 
    print( sum(yard) / len(yard) ) 





def process_subset(dataset_path, subset_name, subset_image_names):
    """
    Process each image subset to create a panorama.

    :param dataset_path: Path to the dataset directory.
    :param subset_name: Name of the current subset.
    :param subset_image_names: List of image filenames in the current subset.
    """
    panorama = initialize_panorama( subset_name, subset_image_names)
    homographies = compute_homographies( subset_name, subset_image_names)

    if homographies:
        panorama = create_panorama(panorama, subset_name, subset_image_names, homographies)
        finalize_and_store_panorama(panorama, subset_name)

def initialize_panorama(subset_name, image_names):
    """ Initialize the panorama with the first image in the subset. """
    return cv2.imread(os.path.join(DATASET_NAME, subset_name, image_names[0]))

def compute_homographies(subset_name, image_names):
    """ Compute homographies between consecutive images in a subset. """
    homographies = []
    feature_points_plot = None  # Initialize feature points plot

    for i in range(len(image_names) - 1):
        current_image, next_image = read_consecutive_images(subset_name, image_names, i)

        # Check if images are loaded correctly
        if current_image is None or next_image is None:
            print(f"Error: Failed to load images at index {i}.")
            continue  # Skip this iteration

        # Convert images to grayscale
        current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        next_image_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

        # Compute homography matrix
        homography_matrix = stitch_images(current_image, current_image_gray, next_image, next_image_gray, feature_points_plot, subset_name)

        if homography_matrix is not None:
            homographies.append(homography_matrix)
        else:
            print(f"Warning: Homography computation failed at index {i}.")
            # Decide how to handle this - skip or break
            # break  # Uncomment to stop processing further if homography fails
            # continue  # Uncomment to skip and continue with the next images

    return homographies


def read_consecutive_images(subset_name, image_names, index):
    """ Read two consecutive images. """
    current_image_path = os.path.join(DATASET_NAME, subset_name, image_names[index])
    next_image_path = os.path.join(DATASET_NAME, subset_name, image_names[index + 1])
    return cv2.imread(current_image_path), cv2.imread(next_image_path)
       


def stitch_images(cur_image, cur_image_gray, next_image, next_image_gray, feature_points_plot,subset_name):

    # Feature extraction
    cur_feature_pts, cur_descs, feature_points_plot = extract_features(cur_image, cur_image_gray, subset_name, feature_points_plot)
    next_feature_pts, next_descs, feature_points_plot = extract_features(next_image, next_image_gray, subset_name ,feature_points_plot)

    # Feature matching
    matches = feature_matching(cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs,subset_name)

    # Find Homography matrix
    if matches is not None and matches.size != 0 and len(matches[:, 0]) >= 4:
        match_pairs_list = []
        for match in matches[:, 0]:
            (x1, y1) = next_feature_pts[match.queryIdx].pt
            (x2, y2) = cur_feature_pts[match.trainIdx].pt
            match_pairs_list.append([x1, y1, x2, y2])

        match_pairs_matrix = np.matrix(match_pairs_list)

        # Run RANSAC algorithm
        H = execute_RANSAC(match_pairs_matrix)
        
        return H

    else:
        print("Can not find enough key points.")
        return None

    

def create_panorama(panorama_base, image_subset_name, image_filenames, transformation_matrices):
    """
    Apply transformations to images and update the base panorama image.

    :param panorama_base: The base image for the panorama.
    :param image_subset_name: The subset name in the dataset for the images.
    :param image_filenames: List of filenames of images to be merged.
    :param transformation_matrices: Homography matrices for image transformations.
    :return: Updated panorama image.
    """

    cumulative_transformation = None
    for index, matrix in enumerate(transformation_matrices):
        next_image_path = os.path.join(DATASET_NAME, image_subset_name, image_filenames[index + 1])
        next_image = cv2.imread(next_image_path)
        cumulative_transformation = matrix if cumulative_transformation is None else np.matmul(cumulative_transformation, matrix)
        panorama_base = merge_images(panorama_base, next_image, cumulative_transformation,image_subset_name)
    return panorama_base

def finalize_and_store_panorama(panorama, subset_name):
    """
    Enhance the final panorama image, display it, and save it to a file.

    :param panorama: The final panorama image.
    :param subset_name: The subset name used for saving the image.
    """

    enhanced_panorama = enhance_image_brightness(panorama)
    save_panorama(enhanced_panorama, subset_name)

def enhance_image_brightness(image):
    """ Enhance the brightness of an image. """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * BRIGHTNESS_FACTOR, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)




def save_panorama(panorama, subset_name,):
    """
    Save the panorama image to a file within a specific subset folder.

    :param panorama: The panorama image to be saved.
    :param subset_name: The name of the subset corresponding to the panorama.
    :param dataset_name: The name of the dataset directory.
    """
    # Create the path to the subset folder
    subset_folder_path = os.path.join(DATASET_NAME, subset_name)

    # Ensure the subset folder exists
    if not os.path.exists(subset_folder_path):
        os.makedirs(subset_folder_path)

    # Define the file name and save the panorama
    save_path = os.path.join(subset_folder_path, "result.png")
    cv2.imwrite(save_path, panorama)


carmel=[]
fishbowl=[]
goldengate=[]
hotel=[]
yard=[]

def extract_features(image, gray_image, subset_name, existing_feature_plot=None, ):
    """
    Extract feature points from an image using SIFT and plot them.

    :param image: The original image.
    :param gray_image: Grayscale version of the image.
    :param existing_feature_plot: An existing plot with feature points, if any.
    :return: A tuple containing the keypoints, descriptors, and an updated plot with feature points.
    """

     # Extract key points and descriptors using SIFT
    feature_detector = cv2.SIFT_create()

    
    

    start_time = time.time()

    # Your feature detection and computation code here
    keypoints, descriptors = feature_detector.detectAndCompute(gray_image, None)

    end_time = time.time()

    # Calculate the duration in minutes
    
    if (subset_name=="carmel"):
        carmel.append(end_time-start_time)
    if (subset_name=="fishbowl"):
        fishbowl.append(end_time-start_time)

    if (subset_name=="goldengate"):
        goldengate.append(end_time-start_time)

    if (subset_name=="hotel"):
        hotel.append(end_time-start_time)

    if (subset_name=="yard"):
        yard.append(end_time-start_time)


    



    # Draw the keypoints on the original image
    keypoints_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Update the feature points plot
    if existing_feature_plot is None:
        updated_feature_plot = keypoints_image
    else:
        # Merge the existing feature plot with the new keypoints image
        updated_feature_plot = np.concatenate((existing_feature_plot, keypoints_image), axis=1)
        display_feature_points(updated_feature_plot,subset_name)

    return keypoints, descriptors, updated_feature_plot

def display_feature_points(feature_plot, subset_name):
    """
    Display and save an image with feature points for each step.

    :param feature_plot: The plot image with feature points.
    :param subset_name: The name of the subset corresponding to the plot.
    :param dataset_name: The name of the dataset directory.
    :param step_number: The step number in the feature extraction process.
    """
    # Display the feature points plot
    plt.imshow(cv2.cvtColor(feature_plot, cv2.COLOR_BGR2RGB))
    plt.title('Feature Points of ' + subset_name), plt.xticks([]), plt.yticks([])
    
    # Save the plot in the 'feature_extraction' folder
    save_feature_plot(subset_name, )

   

def save_feature_plot(subset_name,):
    """
    Save the feature points plot to a file within a specific subset folder.

    :param subset_name: The name of the subset corresponding to the plot.
    :param dataset_name: The name of the dataset directory.
    """
    # Path to the 'feature_extraction' folder within the subset folder
    feature_extraction_folder = os.path.join(DATASET_NAME, subset_name, "feature_extraction")

    # Ensure the 'feature_extraction' folder exists
    if not os.path.exists(feature_extraction_folder):
        os.makedirs(feature_extraction_folder)

    # Count existing files to determine the new file name
    existing_files = [f for f in os.listdir(feature_extraction_folder) if f.endswith('.png')]
    file_number = len(existing_files) + 1

    # Define the file name using the count and save the plot
    save_path = os.path.join(feature_extraction_folder, f"feature_{file_number}.png")
    plt.savefig(save_path)




def feature_matching(cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs, subset_name):
    """
    Perform feature matching between two images and save the results.

    :param cur_image: The current image.
    :param cur_feature_pts: Feature points of the current image.
    :param cur_descs: Descriptors of the current image.
    :param next_image: The next image.
    :param next_feature_pts: Feature points of the next image.
    :param next_descs: Descriptors of the next image.
    :param subset_name: Name of the subset for saving images.
    """

    # Ensure descriptors are in float32 format
    cur_descs = cur_descs.astype(np.float32) if cur_descs.dtype != np.float32 else cur_descs
    next_descs = next_descs.astype(np.float32) if next_descs.dtype != np.float32 else next_descs

    if (cur_descs is not None and len(cur_descs) > NEAREST_NEIGHBOR_NUM and
            next_descs is not None and len(next_descs) > NEAREST_NEIGHBOR_NUM):

        # Create FLANN matcher and find matches
        flann_matcher = cv2.FlannBasedMatcher()  # Initialize with appropriate parameters
        matches = flann_matcher.knnMatch(next_descs, cur_descs, k=NEAREST_NEIGHBOR_NUM)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append([m])

        # Draw matches and save the image
        if good_matches:
            draw_matches_and_save(cur_image, cur_feature_pts, next_image, next_feature_pts, good_matches, subset_name)
            return np.asarray(good_matches)
        else:
            concatenate_images_and_save(cur_image, next_image, subset_name)
            return None
    else:
        concatenate_images_and_save(cur_image, next_image, subset_name)
        return None 



def save_image(image, subset_name, foldername, filename):
    """
    Save an image to a file within a specific subset folder.

    :param image: The image to be saved.
    :param subset_name: The name of the subset.
    :param dataset_name: The name of the dataset directory.
    :param filename: The filename for saving the image.
    """
    # Path to the specific folder within the subset folder
    folder_path = os.path.join(DATASET_NAME, subset_name,foldername)

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path and save the image
    save_path = os.path.join(folder_path, filename)
    cv2.imwrite(save_path, image)


def draw_matches_and_save(current_image, current_points, next_image, next_points, good_matches, subset_name):
    """ Draw matches on the combined image and save it with a unique name for each call. """
    # Use a static variable to keep track of the number of times the function has been called
    if not hasattr(draw_matches_and_save, "iteration"):
        draw_matches_and_save.iteration = 0  # Initialize the counter on the first call

    match_image = cv2.drawMatchesKnn(current_image, current_points, next_image, next_points, good_matches, None, flags=2)
    colored_match_image = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)

    # Save the image with a unique name using the iteration number
    filename = f"matches_{draw_matches_and_save.iteration}.png"
    save_image(colored_match_image, subset_name, "matches", filename)

    # Increment the iteration counter
    draw_matches_and_save.iteration += 1

    return colored_match_image

def concatenate_images_and_save(current_image, next_image, subset_name):
    """ Concatenate two images for display and save them. """
    combined_image = np.concatenate((current_image, next_image), axis=1)
    colored_combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    save_image(colored_combined_image, subset_name,"concat", "concatenated.png")
    return colored_combined_image


def execute_RANSAC(match_pairs):
    max_inliers = 0
    best_H = None

    for _ in range(RANSAC_ITERATION):
        # Select 4 random matches using a more efficient method
        random_matches = np.vstack([match_pairs[np.random.randint(len(match_pairs))] for _ in range(4)])

        # Compute the homography matrix
        H = find_homography(random_matches)

        # Calculate the number of inliers
        inliers_count = sum(get_geometric_distance(pair, H) < INLIER_THRESH for pair in match_pairs)

        # Update if current inliers are the maximum
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_H = H

        # Exit if threshold is exceeded
        if max_inliers > len(match_pairs) * RANSAC_THRESH:
            break

    return best_H



def merge_images(cur_image, next_image, H,subset_name):

    if not hasattr(merge_images, "iteration"):
        merge_images.iteration = 0  # Initialize the counter on the first call

    next_image_matrix = transpose_and_copy(next_image, "image")
    new_row, new_col = (cur_image.shape[1] + next_image.shape[1], cur_image.shape[0])
    transformed_matrix = np.zeros((new_row, new_col, next_image_matrix.shape[2]))

    # Traverse image pixels to calculate new indices
    for i in range(next_image_matrix.shape[0]):
        for j in range(next_image_matrix.shape[1]):
            dot_product = np.dot(H, [i, j, 1])
            i_match = int(dot_product[0, 0] / dot_product[0, 2] + 0.5)
            j_match = int(dot_product[0, 1] / dot_product[0, 2] + 0.5)
            if 0 <= i_match < new_row and 0 <= j_match < new_col:
                transformed_matrix[i_match, j_match] = next_image_matrix[i, j]

    transformed_next_image = transpose_and_copy(transformed_matrix, "matrix")

    # Brightness enhancement
    brightness_factor = 1  # Adjust this factor to control brightness
    transformed_next_image = np.clip(transformed_next_image * brightness_factor, 0, 255).astype(np.uint8)

    plt.imshow(transformed_next_image)
    plt.title(''), plt.xticks([]), plt.yticks([])
    #plt.show()

    # Find non black pixels in current image and create empty mask
    non_black_mask = np.all(cur_image != [0, 0, 0], axis=-1)
    empty_mask = np.zeros((transformed_next_image.shape[0], transformed_next_image.shape[1]), dtype=bool)
    empty_mask[0:cur_image.shape[0], 0:cur_image.shape[1]] = non_black_mask

    # Assign non black pixels of current image to transformed next image
    transformed_next_image[empty_mask, :] = cur_image[non_black_mask, :]
    transformed_next_image = crop_image(transformed_next_image)

    filename = f"merged_{merge_images.iteration}.png"
    save_image(transformed_next_image, subset_name, "merged", filename)

    # Increment the iteration counter
    merge_images.iteration += 1

    # Save the image with a proper filename extension, e.g., 'merged.png'
   


    plt.imshow(cv2.cvtColor(transformed_next_image, cv2.COLOR_BGR2RGB))
    #plt.show()
    return transformed_next_image


 
def find_homography(match_pairs):
    matrix_list = []
    for pair in match_pairs:
        p1 = np.matrix([pair.item(0), pair.item(1), 1])
        p2 = np.matrix([pair.item(2), pair.item(3), 1])
        matrix_list.append([0, 0, 0,
              -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)])

        matrix_list.append([-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)])

    matrixA = np.matrix(matrix_list)
    u, s, vh = np.linalg.svd(matrixA)
    H = np.reshape(vh[8], (3, 3))
    H = (1 / H.item(8)) * H
    return H


# Calculate the geometric distance of matched points
def get_geometric_distance(match_pairs, H):
    estimated = np.dot(H, np.transpose(np.matrix([match_pairs[0].item(0), match_pairs[0].item(1), 1])))
    # Eliminate division by zero and return inefficient d
    if estimated.item(2) == 0:
        return INLIER_THRESH + 1
    err = np.transpose(np.matrix([match_pairs[0].item(2), match_pairs[0].item(3), 1])) - (1 / estimated.item(2)) * estimated
    return np.linalg.norm(err)


def crop_image(image):
    # Crop top if black
    if not np.sum(image[0]):
        return crop_image(image[1:])

    # Crop bottom if black
    elif not np.sum(image[-1]):
        return crop_image(image[:-2])

    # Crop left if black
    elif not np.sum(image[:, 0]):
        return crop_image(image[:, 1:])

    # Crop right if black
    elif not np.sum(image[:, -1]):
        return crop_image(image[:, :-2])

    return image


def transpose_and_copy(image_data, data_type):
    """
    Transpose and copy image data based on the specified data type.

    This function transposes the given image data by swapping the first two dimensions.
    If the data type is 'matrix', the transposed data is converted to an unsigned 8-bit integer format.

    Parameters:
    image_data (numpy.ndarray): The image data to be transposed and copied.
    data_type (str): The type of the data, expected to be 'matrix' or other.

    Returns:
    numpy.ndarray: The transposed and potentially type-converted copy of the input data.
    """

    # Perform a deep copy and transpose (swap) the first two dimensions
    transposed_data = np.transpose(image_data.copy(), (1, 0, 2))

    # Convert to unsigned 8-bit integer if the data type is 'matrix'
    if data_type == "matrix":
        transposed_data = transposed_data.astype('uint8')

    return transposed_data

if __name__ == '__main__':
    get_subset_names(DATASET_NAME)
    main()
