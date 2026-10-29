import cv2 
import torch
import numpy as np

from picamera2 import Picamera2

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cpu"

model = DepthAnythingV2(
    # encoder is the part of a neural netowrk that processes and image and extracts important visual information.
    #vits means vision Transformer small
    # 64 overlayed feature maps/channels
    encoder='vits',
    features = 64,
    out_channels=[48,96,192,384]
    # four stages of the decoder architechture
    #stage one has 48 channels, stage 2 has 96 channels, stage 3 has 192 channels, stage 4 has 384 channels

)

checkpoint = "checkpoints/depth_anything_v2_vits.pth"

state_dict = torch.load(
    checkpoint,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)

model = model.to(DEVICE).eval()

print("model loaded")


camera = Picamera2()

camera.configure(
    camera.create_preview_configuration(
        main= {
            "size":(320,240),
            "format": "RGB888"
        }
    )
)

camera.start()

print("camera started")
# a pixel which is pure red in RGB would be [255,0,0]
# but that same pixel in BGR would be [0,0,255]

try:
    while True:
        frame = camera.capture_array()
        frame = cv2.cvtColor(
            frame,
            cv2.RGB2BGR
        )

        depth = model.infer_image(frame)
        depth_display = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        depth_display = depth_display.astype(
            np.uint8
        )

        cv2.imshow("Raspberry Pi camera",frame)
        cv2.imshow("Depth analysis",depth_display)


        # press r to exit
        if cv2.waitKey(1) & 0xFF == ord("r"):
            break

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("camera stopped")
