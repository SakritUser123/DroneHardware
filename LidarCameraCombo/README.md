# lidepth — metric depth for every camera pixel (2-D lidar + camera + AI)

**Ground test, no drone yet.** A 360° 2-D lidar (LDRobot **LD19**, or LD06) measures true distances
in one horizontal plane. A Raspberry Pi Camera Module 3 sees the scene, and the AI model
**Depth Anything V2 Small** guesses the *shape* of the depth for every pixel — but only up to an
unknown scale and shift. Where the lidar line crosses the image we know the true distance, so on
every frame we solve

```
1 / Z_metric  =  a · d_model  +  b          (robust fit over ALL lidar points in view)
```

and apply it to the whole image → a metric depth map (metres) for every pixel, plus a confidence
map (high near the lidar line, lower far above/below it).

```
 LD19 lidar ──serial──► driver thread ─► time-stamped points ─┐
                                                               ├─► project into image ─► fit a,b ─► Z(u,v)
 Camera Module 3 ─► undistort ─► Depth Anything V2 (ONNX) ─────┘                (RANSAC + LSQ, EMA)
```

Everything runs on a **Raspberry Pi 4** (live) and also on a **Mac** (offline, from recordings).
A **mock mode** runs the whole thing with a simulated room and lidar — no hardware needed.

---

## Contents
1. [What you need](#1-what-you-need)
2. [Install on the Raspberry Pi](#2-install-on-the-raspberry-pi)
3. [Install on the Mac (offline processing)](#3-install-on-the-mac-offline-processing)
4. [Wire the lidar](#4-wire-the-lidar)
5. [Run the test — step by step](#5-run-the-test--step-by-step)
6. [Outputs explained](#6-outputs-explained)
7. [Speed (FPS) and how to make it faster](#7-speed-fps-and-how-to-make-it-faster)
8. [Troubleshooting](#8-troubleshooting)
9. [Assumptions & hardware notes](#9-assumptions--hardware-notes)
10. [How the code is organised / future drone integration](#10-how-the-code-is-organised--future-drone-integration)

---

## 1. What you need

| Item | Notes |
|---|---|
| Raspberry Pi 4 Model B | Raspberry Pi OS **64-bit Bookworm**, Python 3.11 (comes with it) |
| Raspberry Pi Camera Module 3 (standard lens) | CSI ribbon. **Pi 4 uses the 15-pin cable that comes with the camera.** (A Pi 5 would need the 22-pin-to-15-pin cable.) |
| LDRobot DTOF LD19 kit (FHL-LD19) | with its USB-to-UART adapter board (or use the Pi GPIO UART) |
| Rigid mount | camera directly above (or below) the lidar, both facing forward. 3-D print, aluminium bracket, or hot-glued plywood — it must **not move** after calibration |
| Printed checkerboard | `calib/checkerboard_A4.pdf` (included), taped flat on cardboard |
| Tape measure | for the ground-truth test |
| Screen for the Pi | HDMI monitor, or VNC (Raspberry Pi Connect / RealVNC) — needed for the interactive tools |
| Mac (optional) | for faster offline processing of recordings |

---

## 2. Install on the Raspberry Pi

Open a terminal on the Pi (or `ssh pi@raspberrypi.local`). Copy each block, one at a time.

**2.1 Update the system and install the system packages**
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-yaml \
                    python3-serial python3-matplotlib python3-pytest python3-venv
```
> Picamera2 and OpenCV **must** come from `apt` on the Pi (pip versions don't talk to the camera
> stack / have no GUI). That's why the virtual environment below uses `--system-site-packages`.

**2.2 Check the camera** (power off before plugging the ribbon in; blue side of the ribbon faces the
Ethernet port on a Pi 4)
```bash
rpicam-hello --list-cameras        # should list "imx708" (Camera Module 3)
rpicam-hello -t 5000               # shows a 5-second preview (needs a screen)
```
On Bookworm the camera is detected automatically (`camera_auto_detect=1` in
`/boot/firmware/config.txt`). If it is not listed, see [Troubleshooting](#8-troubleshooting).

**2.3 Allow your user to use serial ports** (then **log out and back in**, or reboot)
```bash
sudo usermod -aG dialout $USER
```

**2.4 Get the project onto the Pi** — copy the folder (USB stick, `scp`, or `git clone`) to e.g.
`~/lidar_cam_depth`, then:
```bash
cd ~/lidar_cam_depth
python3 -m venv --system-site-packages ~/lidepth-venv
source ~/lidepth-venv/bin/activate          # do this in every new terminal
pip install -r requirements-pi.txt          # installs ONNX Runtime (and pymavlink, unused for now)
```

**2.5 Download the depth model** (~99 MB, Apache-2.0 licence)
```bash
python scripts/get_model.py
```
You should see `Model OK: input 'pixel_values' ...` and a test inference time.

**2.6 Self-test without any hardware**
```bash
python -m pytest -q                                   # all tests should pass
python -m lidepth record  --mock --duration 5 --name demo
python -m lidepth process sessions/demo
python -m lidepth evaluate sessions/demo --auto-gt 30
python -m lidepth live --mock                         # window with simulated room (q to quit)
```
If these work, the software is installed correctly. (Mock "accuracy" and mock FPS are **not**
meaningful — the simulator's fake model is much easier than the real one, and drawing the simulated
room is slow.)

---

## 3. Install on the Mac (offline processing)

```bash
cd lidar_cam_depth
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_model.py
python -m pytest -q
```
Copy a recording from the Pi and process it:
```bash
scp -r pi@raspberrypi.local:~/lidar_cam_depth/sessions/2026-10-30_14-03-11 sessions/
scp pi@raspberrypi.local:~/lidar_cam_depth/calib/*.yaml calib/          # the calibration files!
python -m lidepth process sessions/2026-10-30_14-03-11
python -m lidepth evaluate sessions/2026-10-30_14-03-11
```
On the Mac you can use a bigger model input for sharper maps, e.g. `--set model.input_long_side=518`.

---

## 4. Wire the lidar

The LD19 cable has 4 wires (ZH1.5T-4P connector). From the LD19 manual: **Tx** (data out),
**PWM** (speed control), **GND**, **P5V** (5 V, ~0.3 A at start-up).

### Option A — USB adapter board (easiest, recommended)
1. Plug the lidar cable into the small adapter board from the kit.
2. Connect the adapter board to a Pi USB port with the USB cable.
3. The lidar should start spinning. Find the port:
   ```bash
   ls /dev/ttyUSB* /dev/ttyACM*          # usually /dev/ttyUSB0 (CP2102 chip)
   dmesg | tail                           # shows what was just plugged in
   ```
4. `config.yaml` already says `port: /dev/ttyUSB0`. If yours is different, edit it or use
   `--set lidar.port=/dev/ttyACM0`.

### Option B — Pi GPIO UART (no adapter)
```
           Raspberry Pi 4 GPIO header (looking at the board, USB ports at the bottom)
                    3V3  (1) (2)  5V    ◄── lidar P5V  (red)
                  GPIO2  (3) (4)  5V
                  GPIO3  (5) (6)  GND   ◄── lidar GND  (black)
                  GPIO4  (7) (8)  GPIO14 / TXD   (not used — the lidar only sends)
                    GND  (9) (10) GPIO15 / RXD   ◄── lidar Tx  (data)
                 GPIO17 (11) (12) GPIO18
                 GPIO27 (13) (14) GND   ◄── lidar PWM  (tie to GND = internal 10 Hz speed control)
```
* Lidar **Tx → Pi pin 10 (GPIO15/RXD)**, **GND → pin 6**, **P5V → pin 2**, **PWM → pin 14 (GND)**.
  The LD19 manual says PWM must be grounded when you don't control the speed externally.
* The LD19's Tx is 3.3 V logic, which is safe for the Pi. *(If your unit's datasheet says 5 V
  logic, put a level shifter / voltage divider on Tx — the Pi's RX pin is 3.3 V only.)*
* Enable the UART **without** the login console:
  ```bash
  sudo raspi-config     # Interface Options → Serial Port →
                        #   "login shell over serial?"  → No
                        #   "serial port hardware enabled?" → Yes
  ```
* On a Pi 4, `/dev/serial0` is the weak "mini-UART" unless Bluetooth is disabled. For a reliable
  230400 baud link add this line to the end of `/boot/firmware/config.txt`, then reboot:
  ```
  dtoverlay=disable-bt
  ```
* Use `--set lidar.port=/dev/serial0` (or edit `config.yaml`).

### Check the lidar
```bash
python -m lidepth lidar-test --seconds 10 --save lidar_top.png
```
Good: `scan 10.0 Hz | ~450 pts/scan | CRC errors 0 (or a handful)`. Hold a box ~1 m in front of
the lidar: *straight ahead (±15°)* should read about 1.0 m. If the box shows up at another angle,
the lidar's 0° direction is not where you think — note the angle; you'll use it as the yaw.

---

## 5. Run the test — step by step

Always `cd ~/lidar_cam_depth && source ~/lidepth-venv/bin/activate` first.

### Step 1 — Fix the focus
The Camera Module 3 has autofocus; the program switches it to **manual** at
`camera.lens_position` (dioptres = 1 / distance in metres; default `1.0` = 1 m, which keeps
roughly 0.5 m – infinity acceptably sharp). Calibration depends on focus, so **never change this
value after calibrating** (if you do, re-run Step 2).

### Step 2 — Calibrate the camera (intrinsics)
1. Print `calib/checkerboard_A4.pdf` at **100 % / actual size** (no "fit to page").
   *(Need a new one? `python -m lidepth make-checkerboard`)*
2. Measure one square with a ruler. If it isn't 25.0 mm, put your value in
   `config.yaml → calibration.square_size_mm`.
3. Tape it perfectly flat onto stiff cardboard.
4. Run (with a screen):
   ```bash
   python -m lidepth calibrate-camera
   ```
   Hold the board in front of the camera. When coloured corner dots appear, press **SPACE** to grab
   a view. Grab **15–25 views**: centre, all four corners of the image, near (fills the image) and
   far, tilted left/right/up/down ~30°. Then press **C**.
   *Headless alternative:* `python -m lidepth calibrate-camera --auto` (grabs a view every 2 s
   while it sees the board — keep moving it).
5. **Good result:** `RMS reproj error < 0.5 px` (OK < 1.0). `cx, cy` close to the image centre.
   Field of view about **53° × 41°** at 640×480 (see §9 for why it isn't 66°).
   Saved to `calib/camera.yaml`. If the error is high, delete the "worst views" listed from
   `calib/camera_images/` and re-run with `--images calib/camera_images`.

### Step 3 — Mount the sensors rigidly
* Camera **directly above** the lidar (or below), both facing the same way, camera roughly level.
  5–10 cm vertical gap is ideal: small enough that the lidar line is near the image centre.
* Make sure the camera doesn't see the lidar's own body and the lidar's view forward isn't blocked
  by the camera bracket.
* Measure with a ruler and put the values in `config.yaml → extrinsics_default`:
  `cam_up_m` = height of the camera **lens centre** above the lidar's **scan plane** (the middle of
  the spinning window). Negative if the camera is below. Leave the others at 0 for now.

### Step 4 — Calibrate lidar → camera alignment (extrinsics)
Set up a scene with **clear vertical edges at different distances** 1–4 m away: a door frame, a
box on a chair at lidar height, a vertical pole/broom handle. Then:
```bash
python -m lidepth calibrate-extrinsics
```
You see the camera image with coloured lidar dots (red = near, blue = far), and a top view on the
right with the camera's field of view (yellow wedge).
1. **Yaw first** (`a`/`d`): the dots on the pole must sit exactly on the pole; the distance jumps
   (colour changes) must line up with the box/door edges. If nothing lines up at all, look at the
   top view and turn the yellow wedge towards the objects in front of the camera.
2. **Pitch** (`w`/`s`) and **height** (`r`/`f`): the row of dots should cross objects at the
   height of the lidar's scan plane. Hold a flat box edge exactly at lidar height and match it.
3. **Roll** (`z`/`x`): if dots line up on the left side but not the right, the line is tilted.
4. `[` / `]` change the step size; `e` shows edges only (makes alignment easier); `0` resets.
5. Press **`p`** to save to `calib/extrinsics.yaml`. **Good:** dots stay on edges at near *and*
   far distances, across the whole image width.

> You can also tune on a recording, e.g. on the Mac:
> `python -m lidepth calibrate-extrinsics --session sessions/<name>` (`n`/`b` = next/prev frame).

### Step 5 — Look at it live
```bash
python -m lidepth live
```
Left: camera + lidar dots (white ring = point used in the fit). Right: metric depth with a
legend in metres; the white line is where the lidar scan plane is. Bottom: FPS, model time,
`a`, `b`, inliers, and the fit error. **Good:** status `FIT` (green), inliers > 70 % of points,
`rel.err` < 5 %. Press `s` to save a snapshot, `q` to quit.
Headless/SSH: `python -m lidepth live --no-display --frames 100` prints the FPS.

### Step 6 — Record a test session
Place objects at **known distances** 0.5–5 m: boxes on the floor, a chair, a wall, some at lidar
height and some **well above / below** it (to test how far the fit extends). Mark the floor with
tape so you can measure later.
```bash
python -m lidepth record --duration 20          # or Ctrl+C to stop; --fps 5 default
```
Keep the rig **still** (or move slowly). Output: `sessions/<date_time>/`.

### Step 7 — Process
```bash
python -m lidepth process sessions/<date_time>
```
On the Pi or on the Mac. Writes depth maps, visualisations and a fit log (see §6).
**Good:** `frames_with_good_fit` close to `frames`, `median_fit_rel_rmse_pct` < 5.

### Step 8 — Evaluate against the tape measure
```bash
python -m lidepth evaluate sessions/<date_time>
```
Click on an object, type its **tape-measured distance in cm from the camera lens** (straight
line), press ENTER. Do 10–30 points over several frames (`n`/`b`), some on the lidar line (yellow),
many above and below it. Press `q`. It prints:

```
group               n valid   MAE cm  medAE cm  mean %  med %   <=5%  <=10%
near lidar line    12    12      4.1       3.0     2.1    1.6    92%   100%
far from line      18    17     21.0      14.2     8.8    6.5    41%    76%
```
and saves `results.csv`, `summary.csv`, `evaluation.png` in `processed/evaluation/`.
(Those numbers are only an example of the *kind* of result to expect: near the line the error
should be close to the lidar's own accuracy, a few %; far above/below it the error depends on the
AI model and is typically several times larger.) Headless: write a CSV `frame,u,v,distance_cm` and
use `--points-csv my_points.csv`. If you measured only the forward distance (not the straight line
from the lens), set `evaluate.gt_type: z`.

---

## 6. Outputs explained

```
sessions/<name>/
  meta.yaml, frames.csv, frames/*.jpg, lidar_packets.bin    ← the recording (raw, re-processable)
  camera.yaml, extrinsics.yaml                              ← calibration in use while recording
  processed/
    depth/000001.npy        float32 (H,W) metric depth Z in metres, along the camera axis; NaN = unknown
    confidence/000001.npz   confidence (0..1) per pixel + the lidar line row per column
    viz/000001.jpg          picture like the live view
    fit_log.csv             per frame: status, a, b (smoothed), a_frame/b_frame (this frame only),
                            n_points, n_inliers, rel_rmse, mae_m, reason, timings
    camera_used.yaml        fx, fy, cx, cy of the undistorted images the depth maps refer to
    summary.yaml            overall stats and processing FPS
    evaluation/             results.csv, summary.csv, evaluation.png
```
Load a depth map in Python: `z = np.load("processed/depth/000001.npy"); print(z[240, 320])`.
`process` uses the **current** `calib/*.yaml`, so you can re-calibrate and re-process old
recordings (`--session-calib` uses the copy saved with the recording instead).

---

## 7. Speed (FPS) and how to make it faster

Almost all the time is the depth model. The target is **≥ 2 FPS on the Pi 4** at the default
settings (`model.input_long_side: 182` → network input 182×140). **Measure it on your Pi:**
```bash
python -m lidepth benchmark --sizes 182 252 364      # model-only time per frame
python -m lidepth live --no-display --frames 60      # prints "Measured end-to-end FPS"
```
I could not run the real model on a Pi 4 while writing this, so the defaults are chosen from the
model's size (≈25 M parameters, ViT-S). Rough expectation for a Pi 4 (please replace with your
measured numbers): 182 px ≈ 2–3 FPS, 252 px ≈ 1–2 FPS, 518 px ≈ 0.2 FPS.

Ways to speed up, in order of usefulness:
1. **Smaller model input** — `--set model.input_long_side=154` (or 126). Fusion still works
   because the lidar fixes the scale; you lose fine detail.
2. **int8 model** (~1.5–2× faster on ARM, slightly less accurate):
   `python scripts/get_model.py --variant int8` then
   `--set model.path=models/depth_anything_v2_small_int8.onnx`.
3. **Fixed input size** (lets ONNX Runtime optimise more):
   `pip install onnx && python scripts/get_model.py --fix-shape 182x140` and point `model.path` at
   the `_182x140.onnx` file.
4. Make sure `model.threads: 4`, nothing else heavy is running, and the Pi isn't throttling
   (`vcgencmd get_throttled` should be `0x0`; use a fan/heatsink and the official 5 V 3 A supply).
5. The display costs ~10–30 ms; `--no-display` is fastest.

For more detail (slower), use 252, 364 or 518 — great on the Mac for offline `process`.

---

## 8. Troubleshooting

| Problem | What to do |
|---|---|
| `No data on /dev/ttyUSB0 — check wiring/baud` | Is the lidar spinning? Check `ls /dev/ttyUSB* /dev/ttyACM*` and use the port that exists. GPIO: Tx must go to pin 10 (RXD), serial enabled in raspi-config with the login shell **disabled**, `dtoverlay=disable-bt`. |
| `Cannot open lidar port … Permission denied` | `sudo usermod -aG dialout $USER`, then log out/in. |
| `Bytes are arriving … but no valid packet` / many CRC errors | Wrong baud (must be 230400), or the mini-UART is in use (add `dtoverlay=disable-bt`), or a loose wire. |
| Lidar not spinning (GPIO) | 5 V and GND connected? PWM tied to GND? The lidar needs ~0.3 A at start-up — use a good Pi power supply. |
| `Picamera2 not found` | `sudo apt install python3-picamera2` and create the venv with `--system-site-packages`. |
| `No camera detected` | Power off, reseat the ribbon (contacts towards the HDMI ports on the Pi side), close the latch. `rpicam-hello --list-cameras`. If still missing, add `dtoverlay=imx708` to `/boot/firmware/config.txt` and reboot. |
| numpy / simplejpeg import errors in the venv | `pip install "numpy<2"` inside the venv (the apt camera packages need numpy 1.x). |
| `Depth model not found` | `python scripts/get_model.py`. |
| No window appears / `could not connect to display` | Use the Pi desktop or VNC. Over SSH use `--no-display`, `calibrate-camera --auto`, or tune/evaluate on the Mac with `--session`. |
| Status stays `NONE`, `too few points` | Lidar dots aren't landing in the image → extrinsics yaw is far off (Step 4), or objects are beyond 12 m, or `sync.tolerance_s` too small. Run `lidar-test`. |
| `model values too uniform` | All lidar points hit the same flat wall at the same distance — the fit needs at least two different distances in view. Put a box in front. |
| `no hypothesis with a > 0` | Usually wrong extrinsics (dots on the wrong objects). Re-do Step 4. |
| Fit OK but depth wrong far above/below the line | That's the limit of a single lidar line + AI model; check `evaluation.png`. Keep the lidar line near the image centre, use a bigger `input_long_side`. |
| Dots slide relative to objects when the rig moves | Time sync: try `--set camera.timestamp_offset_s=0.02` (or −0.02) or `lidar.latency_s`; for the ground test keep the rig still. |
| Low FPS | See §7. Check `vcgencmd get_throttled`. |
| Calibration RMS > 1 px | Board not flat, blurry views, wrong `square_size_mm`, or focus changed. |

Add `-v` to any command for detailed debug logs.

---

## 9. Assumptions & hardware notes

* **LD19 vs LD06.** Both use the identical 47-byte protocol: header `0x54`, ver/len `0x2C`,
  speed (°/s), start angle (0.01°), 12 × (distance mm, intensity), end angle, timestamp (ms, wraps
  at 30000), CRC-8 with the official LDRobot table. LDRobot's own SDK (`ldlidar_stl`) uses one
  parser for both. Differences are in the hardware (ranging rate, PWM speed control; the vendor
  listing for the FHL-LD19 quotes 8000 samples/s, the LD19 manual 4500/s). The code never assumes a
  sample rate: each point's angle is interpolated between the packet's start and end angle, and its
  time from the packet's speed field. `lidar.model` is checked, and the optional legacy LD06
  near-range angle correction (`lidar.ld06_angle_correction`) is **off** by default. ArduPilot
  developers have reported a subtle LD19-vs-LD06 difference in their driver; if your unit behaves
  oddly, check `lidar-test` (CRC errors, scan rate) and tell me what you see.
* **Rotation direction:** clockwise seen from above (angles increase clockwise). Converted to a
  right-handed frame x = forward (lidar 0°), y = left, z = up. Mounted upside-down? Set
  `lidar.upside_down: true`.
* **Lidar 0° direction:** found empirically with `lidar-test` and absorbed into the extrinsic yaw,
  so you don't need to know it in advance.
* **Timestamps:** every lidar packet is stamped with `time.monotonic_ns()` when its last byte
  arrives (back-dated for serial transfer time); each of the 12 points is back-dated by
  (11−i) × angular step / rotation speed. Camera frames use libcamera's `SensorTimestamp`, which on
  Linux is the same `CLOCK_MONOTONIC` clock, plus half the exposure. If they ever disagree by > 2 s,
  the code warns and uses arrival time. Fine-tune with `lidar.latency_s` / `camera.timestamp_offset_s`.
* **640×480 is 4:3, the sensor is 16:9**, so the image is cropped at the sides: horizontal FOV
  ≈ 53°, not 66°. For the full 66° use `camera.width: 640, height: 360` (then re-calibrate).
  `camera.sensor_output_size: [2304, 1296]` forces the full-FOV binned sensor mode.
* **Depth definition:** `Z` = distance along the camera's optical axis (what the fit uses).
  `evaluate` converts to straight-line distance from the lens when `gt_type: range`.
* **Without `calib/camera.yaml`** a rough guess from the datasheet FOV is used (with a warning) so
  that mock/first tests run — always calibrate for real measurements.
* **Licence:** Depth Anything V2 **Small** is Apache-2.0; the larger variants are non-commercial and
  are not used. The ready-made ONNX file comes from the `onnx-community/depth-anything-v2-small`
  conversion on Hugging Face; `scripts/get_model.py --export` makes your own from the official
  PyTorch weights instead.

---

## 10. How the code is organised / future drone integration

```
lidepth/
  lidar/protocol.py   packet format, CRC-8, angle interpolation (wraparound), per-point time, filters
  lidar/buffer.py     thread-safe ring buffer of 360° scans; query points by their own timestamps
  lidar/driver.py     serial background thread + helpful errors
  camera.py           Picamera2 capture, manual focus, monotonic timestamps
  depth_model.py      Depth Anything V2 Small via ONNX Runtime (raw float output, never colours)
  calibration.py      intrinsics (load/save/undistort) and lidar→camera extrinsics
  projection.py       lidar (angle, range) → 3-D → camera → pixel; median sampling
  fusion.py           RANSAC + weighted least squares fit of 1/Z = a·d + b, EMA, fallback, confidence
  pipeline.py         one frame in → metric depth out (used by live/process/tools)
  output/obstacle.py  depth map → 72 × 5° sectors → MAVLink OBSTACLE_DISTANCE   (OFF by default)
  sim.py              simulated room, camera and LD19 (real protocol bytes) for --mock
  session.py          recording format; replays raw packets through the same code as live
  tools/              record, process, live, calibrate-camera, calibrate-extrinsics, evaluate, …
scripts/get_model.py  download / export / quantise the ONNX model
tests/                pytest, no hardware needed
```

**Drone (later, disabled now).** `obstacle_output` in `config.yaml` turns each depth map into the
closest obstacle per 5° sector (only pixels within `band_up_m`/`band_down_m` of the camera height,
optionally a low percentile instead of the minimum) and sends MAVLink `OBSTACLE_DISTANCE` over
`mavlink_port` at `mavlink_baud`. Sectors the camera can't see are sent as "unknown" (65535).
On the flight controller: `PRX1_TYPE = 2` (MAVLink), and `SERIALx_PROTOCOL = 2` on the port wired to
the Pi. It's assumed that lidar 0° = vehicle forward. Enable with
`--set obstacle_output.enabled=true` only when you get to the drone.
