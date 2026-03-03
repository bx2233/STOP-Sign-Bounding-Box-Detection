README.txt
Author: Beibei Xian 
Uni:bx2233

This file describes the parameters used in the cv2.kmeans() function inside my get_box() function, explains why I chose the values I did, and lists some cases where the algorithm might fail.

---------------------------------------------------------------------
1. cv2.kmeans() parameters in my final script
---------------------------------------------------------------------

In my code, the call to cv2.kmeans() looks like this:

    compactness, labels, centers = cv2.kmeans(
        Z, K, None, criteria, attempts, flags
    )

What each parameter means and how I set it:

---------------------------------------------------------------------
1.1 Z – the input feature vectors
---------------------------------------------------------------------
Z is a 2D numpy array where each row corresponds to one pixel, and each column is a feature. I built it by stacking:
- HSV values (H, S, V) of the pixel,
- weighted normalized x and y coordinates (spatial_w times (x_norm, y_norm)).

The spatial weight controls how much the pixel's position matters during clustering. I set spatial_w to 20.0 for normal/bright images and 15.0 for dark images (mean_v < 60). A higher spatial weight makes clusters more compact in image space (good for separating objects that are far apart) but may cause the sign to be split if it's large and the weight is too high. A lower weight lets the sign merge into one cluster even if its parts are spread out, which helps in dark images where color is less reliable.

---------------------------------------------------------------------
1.2 K – number of clusters
---------------------------------------------------------------------
K is the number of clusters the algorithm will try to find. I set it to:
- K = 5 for dark images (mean_v < 60)
- K = 4 for normal/bright images

Dark images have less color information, so more clusters help separate the sign from background noise. For normal images, 4 clusters are usually enough. Using too few clusters could merge the sign with the background; too many clusters might split the sign into tiny pieces. I tested both and found these values work best across the pictures.

---------------------------------------------------------------------
1.3 None – bestLabels (initial labels)
---------------------------------------------------------------------
I pass None here because I want OpenCV to initialize the cluster centers automatically. 

---------------------------------------------------------------------
1.4 criteria – termination criteria
---------------------------------------------------------------------
criteria is a tuple that tells OpenCV when to stop iterating:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)

- The first part means stop when either the maximum number of iterations is reached OR the cluster centers move less than epsilon.
- 25 is the maximum number of iterations. I chose 25 because the algorithm usually converges well before that, and more iterations would just waste time.
- 1.0 is epsilon – the required accuracy. A smaller epsilon would give more precise clusters but take longer.

---------------------------------------------------------------------
1.5 attempts – number of times to run k‑means with different initializations
---------------------------------------------------------------------
attempts = 10. K‑means is sensitive to the initial random placement of centers. By running it 10 times with different initial guesses (using KMEANS_PP_CENTERS for the first attempt of each run), OpenCV picks the result with the best compactness. This makes the final clustering more stable and less dependent on luck. I found 10 attempts gives reliable results without making the code too slow.

---------------------------------------------------------------------
1.6 flags – initialization method
---------------------------------------------------------------------
flags = cv2.KMEANS_PP_CENTERS. This tells OpenCV to use the k‑means++ algorithm for the initial center selection. K‑means++ spreads out the initial centers, which usually leads to better and faster convergence than random initialization. I chose this because it's the recommended method and it worked well in my tests.

---------------------------------------------------------------------
2. Situations where performance may falter significantly
---------------------------------------------------------------------

Even though my algorithm works on most of the 24 images, there are a few situations where it could fail:

- **Extremely dark images where the stop sign is less visible.**  
  The night image fallback uses Otsu thresholding on the enhanced V channel, but if the sign is very dark (less contrast), even that may not find it. In that case the algorithm might return the whole image or a wrong region.

- **Images where the stop sign is severely occluded or cut off.**  
  The algorithm relies on the sign having a roughly square shape and a distinct red color. If some part the sign is hidden (e.g., in this situation by lighting), the remaining red fragment might be too irregular to be selected.

- **Scenes with multiple red objects that are similar in texture and position to the stop sign.**  
  For example, a smaller red sign is right below the sign could confuse the component scoring. The algorithm includes penalties for low vertical position and texture variance, but if the smaller red sign is also high up and has a smooth red surface, it might confuse the algorithm.

- **Images with very strong color casts (e.g., extreme dark blue tint) that shift the red hue into a non‑red range.**  
  The adaptive red mask is designed to handle moderate lighting changes, but if the white balance is completely off, the red pixels might fall outside the hue bands and be missed.

---------------------------------------------------------------------
End of README