# Lite R-CNN v2/v3 inference visualizations

Run the visualization script on GPU with one shared image for both checkpoints:

```powershell
.\.venv312\Scripts\python.exe scripts\visualize_lite_rcnn_inference.py `
  --image data\coco\images\val2017\000000129756.jpg `
  --device cuda
```

The same image is passed through Lite R-CNN v2 and v3. Outputs are grouped by
input image under `outputs/lite_rcnn_inference/<image-name>/v2_image/` and
`v3_image/`. Each folder contains PNG and SVG versions of the original and
transformed input, backbone stage feature overlays, BiFPN/FPN pyramid level
overlays, RPN objectness/box-delta maps, RPN proposals, RoIAlign crops and
7×7 activations, RoI head embeddings, per-proposal class probabilities,
final detections, and a four-panel pipeline overview. Metadata is written
beside the figures as JSON.

Feature-map panels overlay the per-location mean absolute channel activation
on the input image. They visualize spatial response, not a literal RGB tensor.
RoIAlign panels pair crops from the shared input image with the corresponding
pooled 7×7 activation map. The RPN panel draws the first 50 proposals for
readability; proposal count and model details are recorded in metadata.

Optional arguments can select different weights, configs, labels, or output:

```powershell
.\.venv312\Scripts\python.exe scripts\visualize_lite_rcnn_inference.py `
  --image path\to\image.jpg `
  --v2-weights outputs\coco_4090_v2\best.pth `
  --v3-weights outputs\coco_4090_v3\best.pth `
  --output-dir outputs\lite_rcnn_inference `
  --device cuda
```

Matplotlib is an optional dependency. Install it in the active environment with
`python -m pip install -e ".[visualize]"` if needed.
