import cv2
import os
import time
import numpy as np

def _apply_clahe_if_dark(hsv, mean_v_thresh=60):
    # helps red detection in low-light images.
    v = hsv[:,:,2]                     
    if np.mean(v) < mean_v_thresh:     # check if image is dark
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_enhanced = clahe.apply(v)    # apply contrast 
        hsv[:,:,2] = v_enhanced         
    return hsv

def _adaptive_red_mask(hsv, img_mean_v):
    h = hsv[:,:,0].astype(np.int32)   # hue 
    s = hsv[:,:,1].astype(np.int32)   # saturation 
    v = hsv[:,:,2].astype(np.int32)   # value 

    if img_mean_v < 60:                # very dark 
        hue_red = (h <= 20) | (h >= 160)   # wider red hue range
        bright_red = (s >= 15) & (v >= 10) # saturation
        dark_red   = (s >= 10) & (v >= 5)
    elif img_mean_v < 100:              # med dark
        hue_red = (h <= 12) | (h >= 168)
        bright_red = (s >= 30) & (v >= 20)
        dark_red   = (s >= 20) & (v >= 10)
    else:                               # normal
        hue_red = (h <= 5) | (h >= 175)   # bright red 
        bright_red = (s >= 60) & (v >= 60)
        dark_red   = (s >= 50) & (v >= 30)

    return hue_red & (bright_red | dark_red)

def _redness_score_pixel(h, s, v):
    d0 = np.abs(h - 0)                 # distance to hue 0
    d1 = np.abs(h - 179)               # distance to hue 179
    d = np.minimum(d0, d1)             # the closer red end
    hue_score = 1.0 / (d + 1.0)        # convert 
    return hue_score * (s / 255.0) * (v / 255.0)   # multiply by saturation and value 

def _redness_map(hsv):
    # red intensity.
    h = hsv[:,:,0].astype(np.int32)
    s = hsv[:,:,1].astype(np.int32)
    v = hsv[:,:,2].astype(np.int32)
    return _redness_score_pixel(h, s, v)

def _edge_map(img_gray, median_val):
    # find strong edges 
    low_thresh = int(max(0, 0.33 * median_val))
    high_thresh = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(img_gray, low_thresh, high_thresh)
    return edges

def _night_image_fallback(hsv, gray, mean_v):
    # find bright regions
    v = hsv[:,:,2].astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    v_enhanced = clahe.apply(v) # contrast enhancement

    # otsu's method to separate bright from dark
    _, bright_mask = cv2.threshold(v_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    if num_labels <= 1:          # no components besides background
        return None

    # compute edge map 
    median_val = np.median(gray)
    edges = _edge_map(gray, median_val)

    H, W = hsv.shape[:2]
    img_area = H * W
    best_score = -1
    best_bbox = None

    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area < 1000:           # too small 
            continue
        if area > 0.4 * img_area: # too large 
            continue

        # ratio 
        ar = w / float(h + 1e-6)
        ar_penalty = abs(ar - 1.0) * 0.3

        # edge density 
        comp = (labels == lab)
        edge_density = np.mean(edges[comp]) / 255.0

        score = edge_density * np.log(area + 1) - ar_penalty
        if score > best_score:
            best_score = score
            best_bbox = (x, y, x+w, y+h)

    return best_bbox

def _boundary_edge_strength(mask, edges, dilation_radius=2):
    # high value means the object is defined and separated from background
    kernel = np.ones((dilation_radius*2+1, dilation_radius*2+1), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    ring = cv2.bitwise_xor(dilated, mask.astype(np.uint8))   # ring = dilated - original
    ring_edges = cv2.bitwise_and(ring, edges)                 
    total_ring = np.sum(ring)
    if total_ring == 0:
        return 0
    return np.sum(ring_edges) / total_ring

def _solidity(mask):
    # solidity = area / convex hull area
    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea) # largest contour
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt) # convex hull of the contour
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area

def _best_component_bbox(mask, red_mask, hsv, gray, edges):
    # features
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape[:2]
    img_area = float(H * W)
    best = None
    best_score = -1e9

    # the colargest component area 
    max_area = 0
    for lab in range(1, num_labels):
        area = stats[lab][4]
        if area > max_area:
            max_area = area

    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]

        # size filtering
        if area < 1200:
            continue
        area_frac = area / img_area
        if area_frac > 0.35:   # too large 
            continue

        comp = (labels_cc == lab)
        red_density = float(red_mask[comp].mean()) 
        if red_density < 0.08:  # not red enough
            continue

        ar = w / float(h + 1e-6)
        ar_penalty = abs(ar - 1.0) * 1.2

        # vertical bias 
        cy = y + h / 2.0
        y_bias = 1.0 - (cy / H)      

        # bottom penalty 
        bottom = y + h
        bottom_penalty = max(0, (bottom - 0.6*H) / (0.4*H)) if bottom > 0.6*H else 0.0
        bottom_penalty = min(bottom_penalty, 1.0) * 1.5

        # size  
        size_penalty = 0.0
        if max_area > 0:
            size_ratio = area / max_area
            if size_ratio < 0.5:
                size_penalty = (0.5 - size_ratio) * 2.0  

        # border  
        border_penalty = 0.3 if (x == 0 or y == 0 or x+w >= W-1 or y+h >= H-1) else 0.0

        # texture 
        h_vals = hsv[:,:,0][comp]
        s_vals = hsv[:,:,1][comp]
        if len(h_vals) > 0:
            h_mean = np.mean(h_vals)
            h_diff = np.minimum(np.abs(h_vals - h_mean), np.abs(h_vals - (h_mean + 180)))
            h_std = np.std(h_diff)
            s_std = np.std(s_vals)
            var_penalty = min((h_std + s_std) / 50.0, 0.5) * 1.2
        else:
            var_penalty = 0.6

        # edge 
        edge_density = np.mean(edges[comp]) / 255.0 if edges is not None else 0
        edge_bonus = 0.2 * edge_density

        # boundary edge strength 
        comp_mask = (labels_cc == lab).astype(np.uint8) * 255
        boundary_strength = _boundary_edge_strength(comp_mask, edges)
        boundary_bonus = 0.7 * boundary_strength

        # fill ratio 
        roi_red = red_mask[y:y+h, x:x+w].astype(np.uint8)
        fill_ratio = np.sum(roi_red) / float(w * h) if w*h > 0 else 0
        fill_bonus = 0.3 * fill_ratio

        # convex objects have higher score
        solid = _solidity(comp_mask)
        solid_bonus = 0.4 * solid

        # combine 
        score = (2.0 * red_density) + (10.0 * area_frac) + (2.0 * y_bias) \
                - ar_penalty - border_penalty - var_penalty - bottom_penalty - size_penalty \
                + edge_bonus + boundary_bonus + fill_bonus + solid_bonus

        if score > best_score:
            best_score = score
            best = (x, y, x + w, y + h)

    return best

def _expand_using_red_mask(bbox, red_mask):
    # ensure capturing the entire sign 
    xmin, ymin, xmax, ymax = bbox
    H, W = red_mask.shape[:2]

    # how much of the box is red
    current_roi = red_mask[ymin:ymax+1, xmin:xmax+1]
    fill = np.sum(current_roi) / ((xmax-xmin+1)*(ymax-ymin+1)) if (xmax>xmin and ymax>ymin) else 0

    # already filled
    if fill > 0.6:
        return bbox

    # ensure including nearby red
    margin_w = int((xmax - xmin) * 0.1) + 3
    margin_h = int((ymax - ymin) * 0.1) + 3
    x0 = max(0, xmin - margin_w)
    y0 = max(0, ymin - margin_h)
    x1 = min(W-1, xmax + margin_w)
    y1 = min(H-1, ymax + margin_h)

    roi = red_mask[y0:y1+1, x0:x1+1]
    # find red pixels in the enlarged ROI
    pts = np.column_stack(np.where(roi > 0))
    if len(pts) == 0:          # if no red pixels 
        return bbox

    # (row, col) to (x, y) and convex hull
    pts = pts[:, [1, 0]]
    hull = cv2.convexHull(pts)
    x, y, w, h = cv2.boundingRect(hull)  
    # back to original image coordinates
    xmin2 = x0 + x
    ymin2 = y0 + y
    xmax2 = x0 + x + w
    ymax2 = y0 + y + h
    return (xmin2, ymin2, xmax2, ymax2)

def get_box(img):
    if img is None:
        return 0, 0, 0, 0

    H_img, W_img = img.shape[:2]   # image dimensions

    # blur, convert to HSV, grayscale, enhance if dark
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)   
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # CLAHE if the image is dark
    mean_v = np.mean(hsv[:,:,2])
    hsv = _apply_clahe_if_dark(hsv, mean_v_thresh=60)
    mean_v = np.mean(hsv[:,:,2])   

    # relying on bright regions and edges for very dark images
    if mean_v < 40:
        fallback = _night_image_fallback(hsv, gray, mean_v)
        if fallback is not None:
            xmin, ymin, xmax, ymax = fallback
            pad = 2
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(W_img - 1, xmax + pad)
            ymax = min(H_img - 1, ymax + pad)
            return int(xmin), int(ymin), int(xmax), int(ymax)

    median_val = np.median(gray)
    edges = _edge_map(gray, median_val)

    # feature vectors for K-means
    hsv_flat = hsv.reshape((-1, 3)).astype(np.float32)   
    xs = np.tile(np.arange(W_img, dtype=np.float32), H_img)
    ys = np.repeat(np.arange(H_img, dtype=np.float32), W_img)
    xs_norm = (xs / max(W_img - 1, 1)).reshape((-1, 1))   
    ys_norm = (ys / max(H_img - 1, 1)).reshape((-1, 1))

    # weight spatial features
    spatial_w = 20.0 if mean_v >= 60 else 15.0   # lower weight for dark images
    xy_feat = spatial_w * np.hstack([xs_norm, ys_norm]).astype(np.float32)

    # matrix
    Z = np.hstack([hsv_flat, xy_feat]).astype(np.float32)

    #cluster selection and expansion
    red_mask_full = _adaptive_red_mask(hsv, mean_v)               
    red_mask_flat = red_mask_full.reshape(-1)                     

    # k-means clustering
    K = 5 if mean_v < 60 else 4           
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(Z, K, None, criteria, attempts, flags)
    labels = labels.flatten()   

    # score each cluster 
    cluster_data = []
    for k in range(K):
        idx = (labels == k)
        n = idx.sum()
        if n == 0:
            continue

        h_k = centers[k, 0]
        s_k = centers[k, 1]
        v_k = centers[k, 2]
        center_red = _redness_score_pixel(np.array([h_k]), np.array([s_k]), np.array([v_k]))[0]

        pixel_red = float(red_mask_flat[idx].mean())

        # dark images trust center redness more
        if mean_v < 60:
            combined = 0.8 * center_red + 0.2 * pixel_red
        else:
            combined = 0.5 * center_red + 0.5 * pixel_red

        area_frac = n / float(H_img * W_img)  
        cluster_data.append((combined, center_red, pixel_red, area_frac, k))

    # clusters by combined score
    cluster_data.sort(reverse=True, key=lambda x: x[0])

    # top cluster
    selected = [cluster_data[0][4]]
    if len(cluster_data) > 1 and cluster_data[1][0] >= 0.7 * cluster_data[0][0]:
        selected.append(cluster_data[1][4])

    # build a mask 
    mask = np.zeros((H_img * W_img,), dtype=np.uint8)
    for k in selected:
        mask[labels == k] = 255
    mask = mask.reshape((H_img, W_img))

    # keep red 
    mask = cv2.bitwise_and(mask, red_mask_full.astype(np.uint8) * 255)

    # clean and connect parts
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  

    # dilate then erode 
    kernel_dilate = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    mask = cv2.erode(mask, kernel_dilate, iterations=1)

    # bounding box 
    bbox = _best_component_bbox(mask, red_mask_full, hsv, gray, edges)

    if bbox is None:
        if mean_v < 60:
            fallback = _night_image_fallback(hsv, gray, mean_v)
            if fallback is not None:
                xmin, ymin, xmax, ymax = fallback
                pad = 2
                xmin = max(0, xmin - pad)
                ymin = max(0, ymin - pad)
                xmax = min(W_img - 1, xmax + pad)
                ymax = min(H_img - 1, ymax + pad)
                return int(xmin), int(ymin), int(xmax), int(ymax)
        return 0, 0, W_img - 1, H_img - 1

    xmin, ymin, xmax, ymax = bbox
    box_area = (xmax - xmin) * (ymax - ymin)

    # try night fallback again
    if box_area > 0.45 * (H_img * W_img) or box_area < 2000:
        if mean_v < 60:
            fallback = _night_image_fallback(hsv, gray, mean_v)
            if fallback is not None:
                fxmin, fymin, fxmax, fymax = fallback
                fbox_area = (fxmax - fxmin) * (fymax - fymin)
                if fbox_area < 0.5 * (H_img * W_img) and fbox_area > 800:
                    xmin, ymin, xmax, ymax = fxmin, fymin, fxmax, fymax

    # expand to ensure the full sign
    xmin, ymin, xmax, ymax = _expand_using_red_mask((xmin, ymin, xmax, ymax), red_mask_full)

    # avoid cutting off the sign edges
    pad = 2
    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = min(W_img - 1, xmax + pad)
    ymax = min(H_img - 1, ymax + pad)

    return int(xmin), int(ymin), int(xmax), int(ymax)


if __name__ == "__main__":
    start_time = time.time()
    dir_path = './images/'
    for i in range(1, 25):
        img_name = f'stop{i}.png'
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        xmin, ymin, xmax, ymax = get_box(img)
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        output_path = f'./results/{img_name}'
        cv2.imwrite(output_path, img)
    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds")
