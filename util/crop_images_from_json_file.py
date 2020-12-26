import json
from PIL import Image
import os


def load_annotation_from_json_file(json_file_path):
    with open(json_file_path, 'r') as f:
        content = json.load(f)
    return content


def crop_and_save_images(pil_image, x1, y1, x2, y2, crop_path):
    crop_image = pil_image.crop((x1, y1, x2, y2))
    crop_image.save(crop_path)


def get_position(shape_attributes):
    x_tl = shape_attributes['x']
    y_tl = shape_attributes['y']
    width = shape_attributes['width']
    height = shape_attributes['height']
    x_rb = x_tl + width
    y_rb = y_tl + height
    return x_tl, y_tl, x_rb, y_rb


def main():
    json_file_path = '/home/may_0/Downloads/annotation.json'
    saved_dir = '/home/may_0/Downloads/test_bill_images'
    image_dir = '/home/may_0/Downloads/hoadontiendien'
    mapping_file = '/home/may_0/Downloads/hoadontiendien/mapping.txt'
    annotation_dict = load_annotation_from_json_file(json_file_path)
    count = 0
    for key, value in annotation_dict.items():
        image_name = value["filename"]
        regions_list = value['regions']
        if len(regions_list) == 0:
            continue
        image_path = os.path.join(image_dir, image_name)
        pil_image = Image.open(image_path)
        for region in regions_list:
            shape_attributes = region["shape_attributes"]
            label = region["region_attributes"]["label"]
            x1, y1, x2, y2 = get_position(shape_attributes)
            crop_image_path = os.path.join(saved_dir, f'image_{count}.jpg')
            crop_and_save_images(pil_image, x1, y1, x2, y2, crop_image_path)
            count = count + 1
            saved_string = f'image_{count}.jpg\t{label}\n'
            with open(mapping_file, 'a', encoding="utf-8") as f:
                f.write(saved_string)

main()