# Clock Detection from Video, Image, or Webcam

This project uses **OpenCV** and **NumPy** to detect and extract the time from clocks visible in images, videos, or live webcam streams.

## Features

- Detect circular clock faces using Hough Circle Transform.
- Identify clock hands and estimate the time using Hough Line Transform.
- Supports input from:
  - Image files
  - Video files
  - Webcam feeds.

---

## Installation

First, clone this repository and install the required dependencies.

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
opencv-python-headless==4.8.1.78
numpy==1.26.0
```

---

## Usage

Use the following command-line arguments to run the program:

- **`--mode`**: Select the input mode: `video`, `image`, or `webcam`.
- **`--path`**: Path to the video or image file (required for `video` and `image` modes).

### Example Commands

#### Detect time from an image:

```bash
python clock_detection.py --mode image --path path/to/image.jpg
```

#### Detect time from a video:

```bash
python clock_detection.py --mode video --path path/to/video.mp4
```

#### Detect time from a webcam:

```bash
python clock_detection.py --mode webcam
```

---

## How It Works

1. **Circle Detection:** The clock face is detected using the Hough Circle Transform.
2. **Clock Hand Detection:** Lines are identified within the clock face using the Hough Line Transform.
3. **Time Calculation:** The angles of the hour, minute, and second hands are analyzed to determine the time.
4. **Display:** The time is displayed on the video frame or webcam feed.

---

## Keyboard Controls

- **Press `q`**: Exit the video stream or webcam feed.

---

## Troubleshooting

- Ensure the image or video contains a clear, unobstructed clock face.
- For webcams, verify permissions if the stream fails to start.
- Double-check the file path for video or image inputs.


---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- **OpenCV** for image processing
- **NumPy** for numerical computations

---

## Contributing

Feel free to fork the repository, make improvements, and submit a pull request!
