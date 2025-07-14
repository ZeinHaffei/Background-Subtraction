# Video Background Subtraction Benchmark

A Python tool for benchmarking background subtraction algorithms on video files using OpenCV.\
Supports KNN, MOG, MOG2, GMG, and manual frame-differencing methods.\
Prints the average processing time per frame for quick algorithm comparisons.

---

## Features

- **Benchmarking:** Measure the average time per frame for various background subtraction methods.
- **Algorithm Options:** Supports KNN, MOG, MOG2, GMG (OpenCV), and a “manual” frame-differencing mode.
- **Command-line interface:** Run from the terminal and specify all options via arguments.

---

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/ZeinHaffei/video-bgsub-benchmark.git
   cd video-bgsub-benchmark
   ```

2. Install dependencies:

   ```bash
   pip install opencv-contrib-python
   ```

---

## Usage

```bash
python your_script.py <video_path> <subtractor_type>
```

- `video_path`: Path to your video file (e.g. `video.mp4`)
- `subtractor_type`: One of: `KNN`, `MOG`, `MOG2`, `GMG`, `MANUAL`

### Examples

**Run with KNN background subtraction:**

```bash
python your_script.py sample_video.mp4 KNN
```

**Run in manual (frame-differencing) mode:**

```bash
python your_script.py sample_video.mp4 MANUAL
```

**Use MOG2:**

```bash
python your_script.py sample_video.mp4 MOG2
```

---

## Output

- Prints the **average processing time per frame** (in seconds) to the console.

---

## Notes

- To see the video frames and the computed foreground mask, uncomment these lines in the script:
  ```python
  # cv2.imshow('Foreground Mask', foreground_mask)
  # cv2.imshow('Video', frame)
  ```
- The script expects a valid video file path.
- For MOG and GMG, you must install `opencv-contrib-python` (not just `opencv-python`).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

- [Zein Al Haffei](https://github.com/ZeinHaffei)

---

**Feel free to open issues or contribute!**

