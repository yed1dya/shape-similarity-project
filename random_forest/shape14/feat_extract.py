from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import winsound
from scipy.fft import fft
from scipy.spatial import distance_matrix, ConvexHull, Delaunay, procrustes
from scipy.spatial.distance import cdist
from shapely.geometry import LinearRing
from skimage.feature import match_template
from skimage.measure import moments_hu
from skimage.transform import resize

# -------------- CONFIG -------------------
INPUT_CSV = "shape14_contours_20250718_151016.csv"
OUTPUT_CSV = f"shape14_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
IMAGE_SIZE = (200, 200)  # modify if needed
DOWNSAMPLED_POINTS = 64
NN_K = 4
ASSIGNMENT_RADIUS = 15  # for feature 27


# -----------------------------------------

def extract_features_for_image(image_id, contours, image_shape, grouped, ref_centroids):
    features = defaultdict(lambda: np.nan)
    try:
        if len(contours) == 0:
            return features  # skip empty

        # Derived data
        contour_areas = np.array([cv2.contourArea(c) for c in contours])
        contour_perimeters = np.array([cv2.arcLength(c, True) for c in contours])
        centroids = np.array([
            tuple(np.mean(c.reshape(-1, 2), axis=0)) if len(c) > 0 else (0, 0)
            for c in contours
        ])

        h, w = image_shape
        cx, cy = w / 2, h / 2
        centroid_offsets = centroids - np.array([cx, cy])

        # 1
        features["num_contours"] = len(contours)
        print("num_contours", end=", ")

        # 2, 3
        features["avg_area"] = np.mean(contour_areas)
        print("avg_area", end=", ")
        features["std_area"] = np.std(contour_areas)
        print("std_area", end=", ")

        # 4
        if len(centroids) >= NN_K + 1:
            dmat = distance_matrix(centroids, centroids)
            np.fill_diagonal(dmat, np.inf)
            nearest = np.sort(dmat, axis=1)[:, :NN_K]
            features["std_4nn"] = np.std(nearest)
            print("std_4nn", end=", ")

        # 5
        circle_similarities = []
        for c in contours:
            pts = c.reshape(-1, 2)
            centroid = np.mean(pts, axis=0)
            dists = np.linalg.norm(pts - centroid, axis=1)
            r = np.mean(dists)
            mad = np.mean(np.abs(dists - r))
            if r != 0:
                circle_similarities.append(mad / r)
        features["circle_similarity"] = np.mean(circle_similarities)
        print("circle_similarity", end=", ")

        # 6
        features["avg_perimeter"] = np.mean(contour_perimeters)
        print("avg_perimeter", end=", ")
        features["std_perimeter"] = np.std(contour_perimeters)
        print("std_perimeter", end=", ")

        # 7
        compactness = 4 * np.pi * contour_areas / (contour_perimeters ** 2 + 1e-6)
        features["avg_compactness"] = np.mean(compactness)
        print("avg_compactness", end=", ")
        features["std_compactness"] = np.std(compactness)
        print("std_compactness", end=", ")

        # 8
        eccentricities = []
        for c in contours:
            if len(c) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                if ma != 0:
                    eccentricities.append(MA / ma)
        features["avg_eccentricity"] = np.mean(eccentricities)
        print("avg_eccentricity", end=", ")
        features["std_eccentricity"] = np.std(eccentricities)
        print("std_eccentricity", end=", ")

        # 9
        solidities = []
        for c, a in zip(contours, contour_areas):
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidities.append(a / hull_area)
        features["avg_solidity"] = np.mean(solidities)
        print("avg_solidity", end=", ")
        features["std_solidity"] = np.std(solidities)
        print("std_solidity", end=", ")

        # 10
        # defect_counts = []
        # epsilon = 1.0  # Adjust as needed (in pixels)
        # for c in contours:
        #     approx = cv2.approxPolyDP(c, epsilon, True)
        #     if len(c) >= 3:
        #         hull = cv2.convexHull(approx, returnPoints=False)
        #         if hull is not None and len(hull) > 3:
        #             defects = cv2.convexityDefects(approx, hull)
        #             if defects is not None:
        #                 defect_counts.append(np.sum(defects[:, 0, 3]) / 256)
        # features["avg_defects"] = np.mean(defect_counts)
        # print("avg_defects", end=", ")

        # 11
        curvatures = []
        for c in contours:
            pts = c[:, 0, :]
            for i in range(1, len(pts) - 1):
                a = pts[i - 1]
                b = pts[i]
                c_ = pts[i + 1]
                ba = a - b
                bc = c_ - b
                angle = np.arccos(np.clip(
                    np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0
                ))
                curvatures.append(angle)
        features["avg_curvature"] = np.mean(curvatures)
        print("avg_curvature", end=", ")
        features["std_curvature"] = np.std(curvatures)
        print("std_curvature", end=", ")

        # 12
        hu_moments = []
        for c in contours:
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            hu = moments_hu(mask)
            hu_moments.append(hu)
        if hu_moments:
            hu_arr = np.array(hu_moments)
            for i in range(7):
                features[f"hu_{i + 1}"] = np.mean(np.abs(hu_arr[:, i]))
            print("hu", end=", ")

        # 13
        fft_descs = []
        for c in contours:
            pts = c.reshape(-1, 2)
            pts = resize(pts, (DOWNSAMPLED_POINTS, 2), preserve_range=True).astype(np.float32)
            complex_pts = pts[:, 0] + 1j * pts[:, 1]
            fd = fft(complex_pts)
            norm_fd = np.abs(fd[1:11]) / np.abs(fd[1]) if np.abs(fd[1]) != 0 else np.zeros(10)
            fft_descs.append(norm_fd)
        if fft_descs:
            fft_arr = np.array(fft_descs)
            for i in range(fft_arr.shape[1]):
                features[f"fft_{i + 1}"] = np.mean(fft_arr[:, i])
            print("fft", end=", ")

        # 14
        features["offset_x_mean"] = np.mean(centroid_offsets[:, 0])
        features["offset_y_mean"] = np.mean(centroid_offsets[:, 1])
        features["offset_x_std"] = np.std(centroid_offsets[:, 0])
        features["offset_y_std"] = np.std(centroid_offsets[:, 1])
        print("offsets", end=", ")

        # 15
        polar = np.stack([
            np.linalg.norm(centroid_offsets, axis=1),
            np.arctan2(centroid_offsets[:, 1], centroid_offsets[:, 0])
        ], axis=1)
        features["var_radii"] = np.var(polar[:, 0])
        print("var_radii", end=", ")
        sorted_angles = np.sort(polar[:, 1])
        angle_gaps = np.diff(np.append(sorted_angles, sorted_angles[0] + 2 * np.pi))
        features["var_angle_gaps"] = np.var(angle_gaps)
        print("var_angle_gaps", end=", ")

        # 16
        if len(centroids) >= 3:
            hull = ConvexHull(centroids)
            features["centroid_hull_area"] = hull.volume
            print("centroid_hull_area", end=", ")
            features["centroid_hull_perimeter"] = np.sum([
                np.linalg.norm(centroids[i] - centroids[j])
                for i, j in zip(hull.vertices, np.roll(hull.vertices, -1))
            ])
            print("centroid_hull_perimeter", end=", ")

        # 17
        quadrant_counts = [0] * 4
        for x, y in centroids:
            q = (y >= cy) * 2 + (x >= cx)
            quadrant_counts[q] += 1
        for i, q in enumerate(quadrant_counts):
            features[f"quad_{i + 1}"] = q
        print("quad", end=", ")

        # 18
        if len(centroids) > 1:
            dists = distance_matrix(centroids, centroids)
            np.fill_diagonal(dists, np.inf)
            features["mean_1nn"] = np.mean(np.min(dists, axis=1))
            print("mean_1nn", end=", ")

        # 19
        if len(centroids) >= 3:
            tri = Delaunay(centroids)
            neighbors = defaultdict(set)
            for simplex in tri.simplices:
                for i in range(3):
                    a, b = simplex[i], simplex[(i + 1) % 3]
                    neighbors[a].add(b)
                    neighbors[b].add(a)
            degrees = np.array([len(v) for v in neighbors.values()])
            features["avg_degree"] = np.mean(degrees)
            print("avg_degree", end=", ")
            # simple clustering coefficient
            features["avg_clustering"] = np.mean([len(v) / (len(centroids) - 1) for v in neighbors.values()])
            print("avg_clustering", end=", ")

        # 20
        union_area = np.sum(contour_areas)
        all_pts = np.concatenate(contours).reshape(-1, 2)
        x, y, w_box, h_box = cv2.boundingRect(all_pts)
        features["area_ratio"] = union_area / (w_box * h_box + 1e-6)
        print("area_ratio", end=", ")

        # 21
        features["bbox_aspect"] = w_box / (h_box + 1e-6)
        print("bbox_aspect", end=", ")

        # 22
        aspect_ratios = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h != 0:
                aspect_ratios.append(w / h)
        features["std_aspect"] = np.std(aspect_ratios)
        print("std_aspect", end=", ")

        # 23
        # Use dummy binarized mask; replace with real image if available
        bin_mask = np.zeros(image_shape, np.uint8)
        cv2.drawContours(bin_mask, contours, -1, 255, -1)
        horiz_flip = np.fliplr(bin_mask)
        vert_flip = np.flipud(bin_mask)
        features["sym_horiz"] = np.corrcoef(bin_mask.flatten(), horiz_flip.flatten())[0, 1]
        features["sym_vert"] = np.corrcoef(bin_mask.flatten(), vert_flip.flatten())[0, 1]
        print("sym_horiz, sym_vert", end=", ")

        # 24 - Fractal dimension (box-counting)
        box_sizes = np.array([2, 4, 8, 16, 32, 64])
        counts = []
        for size in box_sizes:
            resized = resize(bin_mask.astype(bool), (size, size), anti_aliasing=False)
            counts.append(np.sum(resized > 0))
        counts = np.array(counts)
        valid = counts > 0
        if np.sum(valid) >= 2:
            coeffs = np.polyfit(np.log(box_sizes[valid]), np.log(counts[valid]), 1)
            features["fractal_dim"] = -coeffs[0]
        else:
            features["fractal_dim"] = np.nan  # not enough data for a slope
        print("fractal_dim", end=", ")

        # 25 - Procrustes distance: align to reference centroids
        if (len(centroids) == len(ref_centroids) and len(centroids) > 1
                and np.unique(centroids, axis=0).shape[0] > 1):
            try:
                _, _, d = procrustes(ref_centroids, centroids)
                features["procrustes"] = d
            except Exception:
                features["procrustes"] = np.nan
        else:
            features["procrustes"] = np.nan
            print("procrustes", end=", ")

        # Prepare reference data (only once)
        if not hasattr(main, "ref_centroids") or not hasattr(main, "ref_mask"):
            # print("\nAvailable child_id groups:", list(grouped.groups.keys()))
            ref_group = grouped.get_group(0)
            ref_contours = []
            for _, row in ref_group.iterrows():
                x = list(map(int, row["x_values"].split(",")))
                y = list(map(int, row["y_values"].split(",")))
                if len(x) >= 3:
                    contour = np.array(list(zip(x, y)), dtype=np.int32).reshape(-1, 1, 2)
                    ref_contours.append(contour)

            main.ref_mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.drawContours(main.ref_mask, ref_contours, -1, 255, -1)

            main.ref_centroids = np.array([
                tuple(np.mean(c.reshape(-1, 2), axis=0)) if len(c) > 0 else (0, 0)
                for c in ref_contours
            ])

        # 26 - Template-matching score (NCC max)
        score_map = match_template(bin_mask.astype(np.float32), main.ref_mask.astype(np.float32))
        features["template_match_score"] = np.max(score_map)
        print("template_match_score", end=", ")

        # 27 - Occupancy error: unmatched ref centroids
        unmatched = 0
        for ref_c in main.ref_centroids:
            dists = np.linalg.norm(centroids - ref_c, axis=1)
            if len(dists) == 0 or np.min(dists) > ASSIGNMENT_RADIUS:
                unmatched += 1
        features["occupancy_error"] = unmatched / len(main.ref_centroids)
        print("occupancy_error", end=", ")

        # 28 - Normalized assignment cost
        cost_matrix = distance_matrix(centroids, main.ref_centroids)
        if cost_matrix.size == 0:
            features["assignment_cost"] = 1.0  # all unmatched
        else:
            from scipy.optimize import linear_sum_assignment
            img_diag = np.linalg.norm(image_shape)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_cost = np.sum(cost_matrix[row_ind, col_ind]) / img_diag
            extras = abs(len(centroids) - len(main.ref_centroids))
            features["assignment_cost"] = matched_cost / len(col_ind) + extras
        print("assignment_cost", end=", ")

        # 29
        closed = 0
        for c in contours:
            if len(c) >= 3:
                ring = LinearRing(c[:, 0, :])
                if ring.is_closed:
                    closed += 1
        features["closed_pct"] = closed / len(contours)
        print("closed_pct", end=", ")

        # 30, 31
        diameters = []
        for c in contours:
            hull_pts = cv2.convexHull(c)[:, 0, :]
            pairwise = cdist(hull_pts, hull_pts)
            diameters.append(np.max(pairwise))
        features["avg_diameter"] = np.mean(diameters)
        features["std_diameter"] = np.std(diameters)
        print("avg_diameter, std_diameter", end=", ")

        # 32
        diffs = []
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                mask1 = np.zeros(image_shape, np.uint8)
                mask2 = np.zeros(image_shape, np.uint8)
                cv2.drawContours(mask1, [contours[i]], -1, 255, -1)
                cv2.drawContours(mask2, [contours[j]], -1, 255, -1)
                diffs.append(np.mean(np.abs(mask1.astype(np.float32) - mask2.astype(np.float32))))
        features["avg_pairwise_diff"] = np.mean(diffs)
        print("avg_pairwise_diff", end=".")

    except Exception as e:
        print(f"\nError on {image_id}: {e}")
        winsound.MessageBeep()
        exit(0)

    return features


def main():
    df = pd.read_csv(INPUT_CSV)
    grouped = df.groupby("child_id")
    output_rows = []
    # Get reference contours and centroids
    ref_contours = []
    ref_group = grouped.get_group(0)
    for _, row in ref_group.iterrows():
        x = list(map(int, row["x_values"].split(",")))
        y = list(map(int, row["y_values"].split(",")))
        if len(x) >= 3:
            contour = np.array(list(zip(x, y)), dtype=np.int32).reshape(-1, 1, 2)
            ref_contours.append(contour)

    # Compute centroids for reference image
    ref_centroids = np.array([
        tuple(np.mean(c.reshape(-1, 2), axis=0)) if len(c) > 0 else (0, 0)
        for c in ref_contours
    ])

    total = len(grouped)
    for count, (img_id, group) in enumerate(grouped, 1):
        print(f"\n[{count}/{total}], {img_id}: ", end="")

        contours = []
        for _, row in group.iterrows():
            x = list(map(int, row["x_values"].split(",")))
            y = list(map(int, row["y_values"].split(",")))
            if len(x) >= 3:
                contour = np.array(list(zip(x, y)), dtype=np.int32).reshape(-1, 1, 2)
                contours.append(contour)
        if img_id != 0:
            continue
        feats = extract_features_for_image(img_id, contours, IMAGE_SIZE, grouped, ref_centroids)
        feats["child_id"] = img_id
        output_rows.append(feats)
        print()

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    winsound.Beep(400, 1000)


if __name__ == "__main__":
    main()
