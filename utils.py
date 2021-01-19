import csv
import os
import warnings
from collections import defaultdict

import imagesize
import numpy as np
import skimage.io as io
from PIL import Image
from tqdm import tqdm


def csvread(file):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            csv_f = csv.reader(f)
            data = []
            for row in csv_f:
                data.append(row)
    else:
        data = None

    return data


def csvwrite(data, file):
    with open(file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for d in data:
            writer.writerow(d)


def _url_to_license(licenses, mode="http"):
    # create dict with license urls as
    # mode is either http or https

    # create dict
    licenses_by_url = {}

    for license in licenses:
        # Get URL
        if mode == "https":
            url = "https:" + license["url"][5:]
        else:
            url = license["url"]
        # Add to dict
        licenses_by_url[url] = license

    return licenses_by_url


def _list_to_dict(list_data):
    dict_data = []
    columns = list_data.pop(0)
    for i in range(len(list_data)):
        dict_data.append({columns[j]: list_data[i][j] for j in range(len(columns))})

    return dict_data


def convert_category_annotations(original_category_info, classes):
    categories = []
    num_categories = len(original_category_info)
    for i in range(num_categories):
        if (classes is not None) and (original_category_info[i][0] not in classes):
            continue
        cat = {}
        cat["id"] = i + 1
        cat["name"] = original_category_info[i][1]
        cat["freebase_id"] = original_category_info[i][0]

        categories.append(cat)

    return categories


def convert_image_annotations(
    original_image_metadata,
    original_image_annotations,
    original_image_sizes,
    image_dir,
    categories,
    licenses,
    origin_info=False,
):

    original_image_metadata_dict = _list_to_dict(original_image_metadata)
    original_image_annotations_dict = _list_to_dict(original_image_annotations)

    cats_by_freebase_id = {cat["freebase_id"]: cat for cat in categories}

    if original_image_sizes:
        image_size_dict = {
            x[0]: [int(x[1]), int(x[2])] for x in original_image_sizes[1:]
        }
    else:
        image_size_dict = {}

    # Get dict with license urls
    licenses_by_url_http = _url_to_license(licenses, mode="http")
    licenses_by_url_https = _url_to_license(licenses, mode="https")

    # convert original image annotations to dicts
    pos_img_lvl_anns = defaultdict(list)
    neg_img_lvl_anns = defaultdict(list)
    for ann in original_image_annotations_dict[1:]:
        if ann["LabelName"] not in cats_by_freebase_id:
            continue

        cat_of_ann = cats_by_freebase_id[ann["LabelName"]]["id"]
        if int(ann["Confidence"]) == 1:
            pos_img_lvl_anns[ann["ImageID"]].append(cat_of_ann)
        elif int(ann["Confidence"]) == 0:
            neg_img_lvl_anns[ann["ImageID"]].append(cat_of_ann)

    # Create list
    images = []

    # loop through entries skipping title line
    num_images = len(original_image_metadata_dict)
    for i in tqdm(range(num_images), mininterval=0.5):
        # Select image ID as key
        key = original_image_metadata_dict[i]["ImageID"]

        img_filepath = os.path.join(image_dir, f"{key}.jpg")
        if not os.path.exists(img_filepath):
            continue

        # Copy information
        img = {}
        img["id"] = key
        img["file_name"] = key + ".jpg"
        img["neg_category_ids"] = neg_img_lvl_anns.get(key, [])
        img["pos_category_ids"] = pos_img_lvl_anns.get(key, [])
        if origin_info:
            img["original_url"] = original_image_metadata_dict[i]["OriginalURL"]
            license_url = original_image_metadata_dict[i]["License"]
            # Look up license id
            try:
                img["license"] = licenses_by_url_https[license_url]["id"]
            except:
                img["license"] = licenses_by_url_http[license_url]["id"]

        # Extract height and width
        image_size = image_size_dict.get(key, None)
        if image_size is not None:
            img["width"], img["height"] = image_size
        else:
            img["width"], img["height"] = imagesize.get(img_filepath)

        img["width_original"] = img["width"]
        img["height_original"] = img["height"]

        rotation = original_image_metadata_dict[i]["Rotation"]
        if (len(rotation) > 0) and (float(rotation) != 0.0):
            print(
                f"{key} rotated {rotation} {type(rotation)} {len(rotation)}. Swapping image height/width"
            )

            assert float(rotation) in [90.0, 180.0, 270.0]

            if float(rotation) == 90.0:
                method = Image.ROTATE_90
                img["width"] = img["height_original"]
                img["height"] = img["width_original"]
            elif float(rotation) == 180.0:
                method = Image.ROTATE_180
            elif float(rotation) == 270.0:
                method = Image.ROTATE_270
                img["width"] = img["height_original"]
                img["height"] = img["width_original"]

            img_pil = Image.open(img_filepath)
            img_pil = img_pil.transpose(method)
            img_pil.save(img_filepath)

            img["rotation"] = float(rotation)

        # Add to list of images
        images.append(img)

    return images


def convert_instance_annotations(
    original_annotations, images, categories, start_index=0, classes=None
):

    original_annotations_dict = _list_to_dict(original_annotations)

    imgs = {img["id"]: img for img in images}
    cats = {cat["id"]: cat for cat in categories}
    cats_by_freebase_id = {cat["freebase_id"]: cat for cat in categories}

    annotations = []

    annotated_attributes = [
        attr
        for attr in [
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
        if attr in original_annotations[0]
    ]

    num_instances = len(original_annotations_dict)
    for i in tqdm(range(num_instances), mininterval=0.5):
        # set individual instance id
        # use start_index to separate indices between dataset splits
        key = i + start_index
        csv_line = i
        ann = {}
        ann["id"] = key
        image_id = original_annotations_dict[csv_line]["ImageID"]
        if image_id not in imgs:
            continue
        ann["image_id"] = image_id
        ann["freebase_id"] = original_annotations_dict[csv_line]["LabelName"]
        if (classes is not None) and (ann["freebase_id"] not in classes):
            continue
        ann["category_id"] = cats_by_freebase_id[ann["freebase_id"]]["id"]
        ann["iscrowd"] = False

        xmin0 = float(original_annotations_dict[csv_line]["XMin"])
        ymin0 = float(original_annotations_dict[csv_line]["YMin"])
        xmax0 = float(original_annotations_dict[csv_line]["XMax"])
        ymax0 = float(original_annotations_dict[csv_line]["YMax"])
        w_im = imgs[image_id]["width_original"]
        h_im = imgs[image_id]["height_original"]
        w_bb = xmax0 - xmin0
        h_bb = ymax0 - ymin0
        if "rotation" not in imgs[image_id]:
            xmin = xmin0 * w_im
            ymin = ymin0 * h_im
            xmax = xmax0 * w_im
            ymax = ymax0 * h_im
        elif imgs[image_id]["rotation"] == 90.0:
            xmin = ymin0 * h_im
            ymin = (1.0 - xmin0 - w_bb) * w_im
            xmax = xmin + (h_bb * h_im)
            ymax = ymin + (w_bb * w_im)
        elif imgs[image_id]["rotation"] == 180.0:
            xmin = (1.0 - xmin0 - w_bb) * w_im
            ymin = (1.0 - ymin0 - h_bb) * h_im
            xmax = xmin + (w_bb * w_im)
            ymax = ymin + (h_bb * h_im)
        elif imgs[image_id]["rotation"] == 270.0:
            xmin = (1.0 - ymin0 - h_bb) * h_im
            ymin = xmin0 * w_im
            xmax = xmin + (h_bb * h_im)
            ymax = ymin + (w_bb * w_im)
        else:
            assert False

        dx = xmax - xmin
        dy = ymax - ymin
        ann["bbox"] = [round(a, 2) for a in [xmin, ymin, dx, dy]]
        ann["area"] = round(dx * dy, 2)

        for attribute in annotated_attributes:
            ann[attribute.lower()] = int(original_annotations_dict[csv_line][attribute])

        annotations.append(ann)

    return annotations


def _id_to_rgb(array):
    B = array // 256 ** 2
    rest = array % 256 ** 2
    G = rest // 256
    R = rest % 256
    return np.stack([R, G, B], axis=-1).astype("uint8")


def _get_mask_file(segment, mask_dir):
    name = "{}_{}_{}.png".format(
        segment["ImageID"], segment["LabelName"].replace("/", ""), segment["BoxID"]
    )
    return os.path.join(mask_dir, name)


def _combine_small_on_top(masks):
    combined = np.zeros(shape=masks[0].shape, dtype="uint32")
    sizes = [np.sum(m != 0) for m in masks]
    for idx in np.argsort(sizes)[::-1]:
        mask = masks[idx]
        combined[mask != 0] = mask[mask != 0]
    return combined


def _greedy_combine(masks):
    combined = np.zeros(shape=masks[0].shape, dtype="uint32")
    for maks in maksk:
        combined[mask != 0] = mask[mask != 0]
    return combined


def convert_segmentation_annotations(
    original_segmentations,
    images,
    categories,
    original_mask_dir,
    segmentation_out_dir,
    start_index=0,
):

    original_segmentations_dict = _list_to_dict(original_segmentations)

    if not os.path.isdir(segmentation_out_dir):
        os.mkdir(segmentation_out_dir)

    image_ids = list(np.unique([ann["ImageID"] for ann in original_segmentations_dict]))
    filtered_images = [img for img in images if img["id"] in image_ids]

    imgs = {img["id"]: img for img in filtered_images}
    cats = {cat["id"]: cat for cat in categories}
    cats_by_freebase_id = {cat["freebase_id"]: cat for cat in categories}

    for i in range(len(original_segmentations_dict)):
        original_segmentations_dict[i]["SegmentID"] = i + 1

    img_segment_map = defaultdict(list)
    for segment in original_segmentations_dict:
        img_segment_map[segment["ImageID"]].append(segment)

    annotations = []
    segment_index = 0 + start_index
    for img in tqdm(filtered_images, mininterval=0.5):
        ann = dict()
        ann["file_name"] = img["file_name"]
        ann["image_id"] = img["id"]
        ann["segments_info"] = []
        masks = []
        for segment in img_segment_map[img["id"]]:
            # collect mask
            mask_file = _get_mask_file(segment, original_mask_dir)
            mask = io.imread(mask_file)  # load png
            # exclude empty masks
            if np.max(mask) == 0:
                continue
            mask = mask // 255  # set to [0,1]
            mask = mask * segment["SegmentID"]
            masks.append(mask)

            # collect segment info
            segment_info = {}
            # Compute bbox coordinates
            xmin = float(segment["BoxXMin"]) * img["width"]
            ymin = float(segment["BoxYMin"]) * img["height"]
            xmax = float(segment["BoxXMax"]) * img["width"]
            ymax = float(segment["BoxYMax"]) * img["height"]
            dx = xmax - xmin
            dy = ymax - ymin
            # Fill in annotations
            segment_info["bbox"] = [round(a, 2) for a in [xmin, ymin, dx, dy]]
            segment_info["area"] = round(dx * dy, 2)
            segment_info["category_id"] = (cats_by_freebase_id[segment["LabelName"]],)
            segment_info["id"] = segment_index
            segment_index += 1
            # append
            ann["segments_info"].append(segment_info)

        # combined_binary_mask = sum(masks)
        # Looks like many masks overlap
        # currently managed by greedy combining
        combined_binary_mask = _combine_small_on_top(masks)
        # check if masks overlap. If they do we have a problem
        ids_in_mask = len(np.unique(combined_binary_mask[combined_binary_mask != 0]))
        num_segments = len(img_segment_map[img["id"]])
        if ids_in_mask != num_segments:
            print("Overlapping masks in image {}".format(ann["image_id"]))
            values_in_output = np.unique(
                combined_binary_mask[combined_binary_mask != 0]
            )
            ids_in_segments = [
                segment["SegmentID"] for segment in img_segment_map[img["id"]]
            ]
            not_in_segments = [x for x in values_in_output if x not in ids_in_segments]
            not_in_values = [x for x in ids_in_segments if x not in values_in_output]
            print("Not in segments: {}".format(not_in_segments))
            print("Not in pixel values: {}".format(not_in_values))
            # don't include the annotation into the output
            continue

        combined_rgb_mask = _id_to_rgb(combined_binary_mask)
        out_file = os.path.join(segmentation_out_dir, "{}.png".format(ann["image_id"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(out_file, combined_rgb_mask)

        annotations.append(ann)

    return annotations


def filter_images(images, annotations):
    image_ids = list(np.unique([ann["image_id"] for ann in annotations]))
    filtered_images = [img for img in images if img["id"] in image_ids]
    return filtered_images
