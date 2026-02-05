import os
import json
import tqdm

def update_bbox(bbox, crop_params, resize_size):
    """
    Adapts a bounding box [x, y, w, h] to cropping and resizing.
    
    Args:
        bbox: list [x, y, w, h]
        crop_params: dict {'top': int, 'left': int, 'height': int, 'width': int}
        resize_size: tuple (target_width, target_height)
        
    Returns:
        new_bbox: list [x, y, w, h] or None if the box is fully outside the crop.
    """
    x, y, w, h = bbox
    ct, cl, ch, cw = crop_params['top'], crop_params['left'], crop_params['height'], crop_params['width']
    
    # convert to x1, y1, x2, y2
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # adjust coordinates relative to the crop (subtract top-left)
    x1 -= cl
    y1 -= ct
    x2 -= cl
    y2 -= ct
    
    # clip to the crop boundaries (0, 0, cw, ch)
    # if the object is outside, these will result in an invalid box (x1 >= x2 or y1 >= y2)
    x1 = max(0, min(x1, cw))
    y1 = max(0, min(y1, ch))
    x2 = max(0, min(x2, cw))
    y2 = max(0, min(y2, ch))
    
    # check if box is valid (has area)
    if x2 <= x1 or y2 <= y1:
        return None
        
    # resize scales
    # original cropped size was (cw, ch) -> Resized to resize_size
    scale_x = resize_size[0] / cw
    scale_y = resize_size[1] / ch
    
    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y
    
    # Convert back to xywh
    new_w = x2 - x1
    new_h = y2 - y1
    
    return [x1, y1, new_w, new_h]

def process_annotations(src_dir, dest_dir, crop_params, resize_target):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
    
    for filename in tqdm.tqdm(files, desc="Processing Annotations"):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        with open(src_path, 'r') as f:
            data = json.load(f)
            
        new_annotations = []
        original_annotations = data.get('annotations', [])
        
        for ann in original_annotations:
            bbox = ann['bbox']
            new_bbox = update_bbox(bbox, crop_params, resize_target)
            
            if new_bbox is not None:
                ann['bbox'] = new_bbox
                # Update area if present
                ann['area'] = new_bbox[2] * new_bbox[3]
                new_annotations.append(ann)
                
        # Update the data dictionary
        data['annotations'] = new_annotations
        
        # Also update image dimensions in the 'images' list to the new size
        for img in data.get('images', []):
            img['width'] = resize_target[0]
            img['height'] = resize_target[1]
            
        with open(dest_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":    
    # folder with current annotations
    SRC_ANNOTATIONS_DIR = "data/annotations_original_imgs" 
    
    # folder to save the new cropped/resized annotations
    DEST_ANNOTATIONS_DIR = "data/annotations_ZODCropped"
    
    # From resize.py: CustomCrop(top=800, left=500, height=800, width=2840)
    CROP_PARAMS = {
        'top': 800,
        'left': 500,
        'height': 800,
        'width': 2840
    }
    
    # From resize.py: img_resized = img_cropped.resize((256, 256))
    RESIZE_TARGET = (256, 256)
    
    print(f"Source: {SRC_ANNOTATIONS_DIR}")
    print(f"Destination: {DEST_ANNOTATIONS_DIR}")
    
    process_annotations(SRC_ANNOTATIONS_DIR, DEST_ANNOTATIONS_DIR, CROP_PARAMS, RESIZE_TARGET)
    print("Done!")
