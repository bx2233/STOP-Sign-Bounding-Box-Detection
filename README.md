# STOP Sign Bounding Box Detection
**Author:** Beibei Xian

This file describes the parameters used in the `cv2.kmeans()` function inside the `get_box()` function, explains why those values were chosen, and lists situations where the algorithm may fail.

---

## 1. `cv2.kmeans()` Parameters in My Final Script

The call to `cv2.kmeans()` is:

```python
compactness, labels, centers = cv2.kmeans(
    Z, K, None, criteria, attempts, flags
)
```

Below is what each parameter means and how it is set.

---

### 1.1 `Z` — Input Feature Vectors

`Z` is a 2D NumPy array where each row corresponds to one pixel and each column is a feature. It is built by stacking:

- HSV values (`H`, `S`, `V`) of the pixel
- Weighted, normalized `x` and `y` coordinates (`spatial_w * (x_norm, y_norm)`)

The spatial weight controls how much pixel position matters during clustering.

| Condition | `spatial_w` |
|-----------|-------------|
| Normal or bright images | `20.0` |
| Dark images (`mean_v < 60`) | `15.0` |

A higher spatial weight makes clusters more compact in image space, helping separate objects that are far apart. However, if the weight is too high, it may split the stop sign into multiple parts. A lower weight helps the sign stay in one cluster even if its parts are more spread out — especially useful in dark images where color is less reliable.

---

### 1.2 `K` — Number of Clusters

`K` is the number of clusters the algorithm tries to find.

| Condition | `K` |
|-----------|-----|
| Dark images (`mean_v < 60`) | `5` |
| Normal or bright images | `4` |

Dark images contain less useful color information, so more clusters help separate the sign from background noise. For normal images, 4 clusters are usually sufficient. If `K` is too small, the sign may merge with the background; if too large, the sign may be split into small pieces.

---

### 1.3 `None` — bestLabels

`None` is passed for `bestLabels` so that OpenCV initializes the cluster centers automatically.

---

### 1.4 `criteria` — Termination Criteria

```python
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
```

The algorithm stops when **either** condition is met:

- The maximum number of iterations is reached, **or**
- The cluster centers move less than epsilon

| Parameter | Value | Reason |
|-----------|-------|--------|
| Max iterations | `25` | The algorithm usually converges before this; more iterations would mostly waste time |
| Epsilon | `1.0` | A smaller epsilon would be more precise but slower |

---

### 1.5 `attempts` — Number of Runs

```python
attempts = 10
```

K-means is sensitive to the initial placement of centers. Running it 10 times with different initializations allows OpenCV to choose the result with the best compactness, making the output more stable and less dependent on luck. 10 attempts gives reliable results without making the code too slow.

---

### 1.6 `flags` — Initialization Method

```python
flags = cv2.KMEANS_PP_CENTERS
```

This tells OpenCV to use the **k-means++** method for initial center selection. K-means++ spreads out the starting centers, which usually leads to better and faster convergence than random initialization.

---

## 2. Situations Where Performance May Falter Significantly

Even though the algorithm works on most of the 24 images, there are still several situations where it may fail.

1. **Extremely dark images where the stop sign is barely visible**
   The night-image fallback uses Otsu thresholding on the enhanced `V` channel, but if the sign has very low contrast, even that may not detect it correctly. The algorithm may return the whole image or the wrong region.

2. **Images where the stop sign is severely occluded or cut off**
   The algorithm assumes the sign has a roughly square shape and a distinct red color. If part of the sign is hidden, the remaining visible region may be too irregular to be selected correctly.

3. **Scenes with multiple red objects similar to the stop sign**
   For example, a smaller red sign near the real stop sign may confuse the component scoring process. The algorithm includes penalties for low vertical position and texture variance, but if another red object is also high in the image and has a smooth red surface, it may still be selected by mistake.

4. **Images with strong color casts**
   If the image has an extreme blue tint or another strong color shift, the red color of the stop sign may move outside the expected hue range. The adaptive red mask can handle moderate lighting changes, but very severe color distortion may cause the red pixels to be missed.

---

*End of README*
