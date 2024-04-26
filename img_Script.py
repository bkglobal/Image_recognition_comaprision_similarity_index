# import cv2
# import numpy as np
# from skimage.metrics import mean_squared_error


# img1 = cv2.imread('DEMOSTORE01.png') 
# img2 = cv2.imread('DEMOSTORE02.png')

# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 

# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# mse = mean_squared_error(img1_gray, img2_gray)

# print("MSE value:", mse)

# diff = np.abs(img1_gray - img2_gray) 

# # cv2.imshow('Difference', diff) 
# # cv2.waitKey(0) cv2.destroyAllWindows()




import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error


def mse_def(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def compare_images(image1_path, image2_path, method="mse", max_intensity=100):
  """
  Compares two images using the specified method and returns a similarity score.

  Args:
      image1_path (str): Path to the first image.
      image2_path (str): Path to the second image.
      method (str, optional): Comparison method. Defaults to "mse" (Mean Squared Error).
          Other options include "ssim" (Structural Similarity Index).

  Returns:
      float: Similarity score between 0 (completely different) and 1 (identical).
  """

  # Load images in grayscale for robustness
  image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
  image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

  # Ensure images have the same size for comparison
  if image1.shape != image2.shape:
    # Resize the smaller image to match the larger one
    (h1, w1) = image1.shape
    (h2, w2) = image2.shape
    if h1 < h2 or w1 < w2:
      image1 = cv2.resize(image1, (w2, h2), interpolation=cv2.INTER_AREA)
    else:
      image2 = cv2.resize(image2, (w1, h1), interpolation=cv2.INTER_AREA)

#   img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
#   img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  # Compare images based on the chosen method
  if method == "mse":
    # Mean Squared Error: Lower score indicates higher similarity
    mse = mse_def(image1, image2)
    print('mse is -- ', mse)
    # mse = np.mean((image1 - image2) ** 2)
    similarity = 1 - (mse / (max_intensity**2))  # Normalize based on max intensity
    similarity *= 100  # Convert to percentage
    # similarity = mse  # Normalize score (0-1)
  elif method == "ssim":
    # Structural Similarity Index: Higher score indicates higher similarity
    similarity = ssim(image1, image2)
  else:
    raise ValueError(f"Invalid comparison method: {method}")

  return similarity

if __name__ == "__main__":

  print('Totally differnet..')
  image1_path = "DEMOSTORE03.png"  # Replace with your image paths
  image2_path = "DEMOSTORE02.png"

  similarity_score = compare_images(image1_path, image2_path, method="mse")  # Or "ssim"
  print(f"Similarity Score: {similarity_score:.2f}")


  image1_path = "DEMOSTORE03.png"  # Replace with your image paths
  image2_path = "DEMOSTORE02.png"

  similarity_score = compare_images(image1_path, image2_path, method="ssim")  # Or "ssim"
  print(f"Similarity Score: {similarity_score:.2f}")



  print('Similar images..')
  image1_path = "ss1.png"  # Replace with your image paths
  image2_path = "ss2.png"

  similarity_score = compare_images(image1_path, image2_path, method="mse")  # Or "ssim"
  print(f"Similarity Score: {similarity_score:.2f}")

  image1_path = "ss1.png"  # Replace with your image paths
  image2_path = "ss2.png"

  similarity_score = compare_images(image1_path, image2_path, method="ssim")  # Or "ssim"
  print(f"Similarity Score: {similarity_score:.2f}")



  print('Totally different..')
  image1_path = "ss1.png"  # Replace with your image paths
  image2_path = "ss3.png"

  similarity_score = compare_images(image1_path, image2_path, method="mse")  # Or "ssim"
  print(f"Similarity Score: {similarity_score:.2f}")

  image1_path = "ss1.png"  # Replace with your image paths
  image2_path = "ss3.png"

  similarity_score = compare_images(image1_path, image2_path, method="ssim")  # Or "ssim"
  print(f"Similarity Score: {similarity_score:.2f}")


  # Optional: Visualize similarity (requires OpenCV GUI support)
  # diff = cv2.absdiff(image1, image2)
  # cv2.imshow("Image 1", image1)
  # cv2.imshow("Image 2", image2)
  # cv2.imshow("Difference", diff)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()


