import cv2
import math
import numpy as np
import sys
import torch
from detectron2.layers import paste_masks_in_image
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer, GenericMask

# command-line arguments
if len(sys.argv) < 3:
    print("Usage: %s <torchscript_model> <input_image> [output_image]" % sys.argv[0])
    sys.exit(1)
model_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3] if (len(sys.argv) > 3) else None

with torch.no_grad():
    # load model
    print("\nmodel file\n", model_file)
    model = torch.jit.load(model_file)

    # get device
    device = None
    for b in model.buffers():
        device = b.device
        break
    if device is None:
        raise Exception("No buffers?")
    print("\ndevice\n", device)

    # load image, pad to multiples of 32
    print("\ninput file\n", input_file)
    input_img = cv2.imread(input_file)
    image_height, image_width = input_img.shape[:2]
    image_height_new = int(math.ceil(image_height / 32) * 32)
    image_width_new = int(math.ceil(image_width / 32) * 32)
    channels = 3
    if (image_height_new != image_height) or (image_width_new != image_width):
        tmp_img = np.full((image_height_new, image_width_new, channels), (0, 0, 0), dtype=np.uint8)
        tmp_img[0:0 + image_height, 0:0 + image_width] = input_img
        input_img = tmp_img
        image_height, image_width = input_img.shape[:2]
    assert(image_height % 32 == 0 and image_width % 32 == 0)

    # create input
    inp = torch.as_tensor(input_img.astype("float32")).permute(2, 0, 1)

    # make prediction
    output = model.forward(inp)
    boxes, classes, masks_raw, scores, img_size = output
    mask_img = paste_masks_in_image(masks_raw[:, 0, :, :], boxes, (image_height_new, image_width_new), threshold=0.5)
    masks = [GenericMask(np.asarray(x).astype("uint8"), image_height_new, image_width_new) for x in mask_img]
    print("\n# instances\n", len(scores))
    print("\nclasses\n", classes)
    print("\nscores\n", scores)
    print("\nboxes\n", boxes)
    print("\nmasks\n", masks)

    # visualize
    if output_file is not None:
        inst = Instances(image_size=(image_height, image_width))
        inst.set("pred_boxes", boxes)
        inst.set("pred_masks", mask_img)
        inst.set("scores", scores)
        inst.set("classes", classes)
        visualizer = Visualizer(input_img)
        img_vis = visualizer.draw_instance_predictions(inst)
        cv2.imwrite(output_file, img_vis.get_image())
        print("\noutput file\n", output_file)
