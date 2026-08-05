#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-time object detection program with an interactive UI.

Install dependencies:
    python -m pip install ultralytics opencv-python pillow SpeechRecognition pyaudio

Run directly:
    python realtime_object_detection.py

Common examples:
    python realtime_object_detection.py --camera 1
    python realtime_object_detection.py --conf 0.6 --device cpu

Interface functions:
    1. Real-time detection: Detects objects in the entire camera frame.
    2. ROI detection: Hold left click and drag to detect objects only in a specific region.
    3. Voice Assistant: Placeholder for Siri-like voice command integration.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import threading
from collections import deque
from typing import Any


def load_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Imports required libraries and provides installation prompts if missing."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        import cv2
        from PIL import Image, ImageTk
        from ultralytics import YOLO
        import speech_recognition as sr
        
    except ImportError as exc:
        missing_package = getattr(exc, "name", "Required dependencies")
        print(f"Missing Python package or component: {missing_package}", file=sys.stderr)
        print(
            "Please run: python -m pip install ultralytics opencv-python pillow SpeechRecognition pyaudio",
            file=sys.stderr,
        )
        print(
            "If missing tkinter: Windows Python includes it by default; "
            "Ubuntu/Debian can install it via python3-tk.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    return cv2, YOLO, Image, ImageTk, tk, messagebox, sr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time object detection with interactive UI")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, default is 0; try 1 for external webcams",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt", # UPGRADED from yolo11n.pt for better recognition of living beings/objects
        help="YOLO model name or local weight path, default: yolo11s.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Minimum confidence threshold, range 0~1, default: 0.45",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model input image size; smaller increases speed, default: 640",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera feed width, default: 1280",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera feed height, default: 720",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g., cpu, 0 (first GPU); default is auto-select",
    )
    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("--imgsz, --width, and --height must be greater than 0")
    if args.camera < 0:
        parser.error("--camera cannot be less than 0")

    return args


def open_camera(cv2: Any, camera_index: int, width: int, height: int) -> Any:
    """Opens the camera and sets the desired frame size."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    capture = None

    for backend in backends:
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()

    if capture is None:
        raise RuntimeError(
            f"Unable to open camera index {camera_index}. "
            "Please check camera connections and system permissions, or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class ObjectDetectionApp:
    """Interactive interface for real-time detection and ROI selection."""

    DISPLAY_WIDTH = 960
    DISPLAY_HEIGHT = 540

    def __init__(
        self,
        root: Any,
        args: argparse.Namespace,
        model: Any,
        cv2: Any,
        image_module: Any,
        image_tk_module: Any,
        tk: Any,
        messagebox: Any,
        sr: Any,
    ) -> None:
        self.root = root
        self.args = args
        self.model = model
        self.cv2 = cv2
        self.Image = image_module
        self.ImageTk = image_tk_module
        self.tk = tk
        self.messagebox = messagebox
        self.sr = sr

        self.camera: Any | None = None
        self.running = False
        self.mode: str | None = None
        self.after_id: str | None = None
        self.photo_image: Any | None = None
        self.latest_frame_shape: tuple[int, ...] | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.selection_rectangle: int | None = None
        self.frame_times: deque[float] = deque(maxlen=30)
        self.previous_time = time.perf_counter()
        self.read_failures = 0
        
        self.voice_thread = None
        self.listening = False

        self.predict_options: dict[str, Any] = {
            "conf": args.conf,
            "imgsz": args.imgsz,
            "verbose": False,
        }
        if args.device is not None:
            self.predict_options["device"] = args.device

        self._build_interface()

    def _build_interface(self) -> None:
        """Creates the control area, video preview area, and mouse events."""
        self.root.title("Camera Object Detection & Voice Assistant")
        self.root.configure(bg="#eef2f7")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.stop_detection())
        self.root.bind("q", lambda _event: self.close())
        self.root.bind("Q", lambda _event: self.close())

        main_frame = self.tk.Frame(self.root, bg="#eef2f7", padx=14, pady=14)
        main_frame.pack()

        control_frame = self.tk.Frame(
            main_frame,
            width=220,
            height=self.DISPLAY_HEIGHT,
            bg="white",
            padx=18,
            pady=18,
            highlightthickness=1,
            highlightbackground="#d7dee8",
        )
        control_frame.pack(side=self.tk.LEFT, fill=self.tk.Y, padx=(0, 14))
        control_frame.pack_propagate(False)

        self.tk.Label(
            control_frame,
            text="AI Detection",
            font=("Arial", 18, "bold"),
            bg="white",
            fg="#1f2937",
        ).pack(anchor="w", pady=(0, 6))

        self.tk.Label(
            control_frame,
            text="Select a detection mode",
            font=("Arial", 10),
            bg="white",
            fg="#6b7280",
        ).pack(anchor="w", pady=(0, 22))

        button_options = {
            "font": ("Arial", 11, "bold"),
            "width": 16,
            "height": 2,
            "cursor": "hand2",
            "relief": self.tk.FLAT,
        }

        self.tk.Button(
            control_frame,
            text="Real-Time Detection",
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            command=lambda: self.start_detection("realtime"),
            **button_options,
        ).pack(pady=5)

        self.tk.Button(
            control_frame,
            text="ROI Detection",
            bg="#059669",
            fg="white",
            activebackground="#047857",
            activeforeground="white",
            command=lambda: self.start_detection("roi"),
            **button_options,
        ).pack(pady=5)

        self.reselect_button = self.tk.Button(
            control_frame,
            text="Reselect ROI",
            bg="#f59e0b",
            fg="white",
            activebackground="#d97706",
            activeforeground="white",
            disabledforeground="#9ca3af",
            command=self.reset_roi,
            state=self.tk.DISABLED,
            **button_options,
        )
        self.reselect_button.pack(pady=5)
        
        # New Siri / Voice Assistant Button
        self.voice_button = self.tk.Button(
            control_frame,
            text="Start Siri (Voice)",
            bg="#8b5cf6",
            fg="white",
            activebackground="#7c3aed",
            activeforeground="white",
            command=self.start_voice_assistant,
            **button_options,
        )
        self.voice_button.pack(pady=5)

        self.tk.Button(
            control_frame,
            text="Stop Detection",
            bg="#64748b",
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            command=self.stop_detection,
            **button_options,
        ).pack(pady=(20, 5))

        self.tk.Button(
            control_frame,
            text="Exit",
            bg="#dc2626",
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            command=self.close,
            **button_options,
        ).pack(pady=5)

        self.status_text = self.tk.StringVar(value="Please select a mode")
        self.tk.Label(
            control_frame,
            textvariable=self.status_text,
            font=("Arial", 9),
            bg="white",
            fg="#374151",
            justify=self.tk.LEFT,
            wraplength=180,
        ).pack(side=self.tk.BOTTOM, anchor="w", fill=self.tk.X, pady=(12, 0))

        self.canvas = self.tk.Canvas(
            main_frame,
            width=self.DISPLAY_WIDTH,
            height=self.DISPLAY_HEIGHT,
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#374151",
            cursor="crosshair",
        )
        self.canvas.pack(side=self.tk.RIGHT)
        self.image_item = self.canvas.create_image(0, 0, anchor=self.tk.NW)
        self.canvas.create_text(
            self.DISPLAY_WIDTH // 2,
            self.DISPLAY_HEIGHT // 2,
            text="Click a button on the left to start",
            fill="#cbd5e1",
            font=("Arial", 18),
            tags="placeholder",
        )

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

    def start_voice_assistant(self) -> None:
        """Initializes the voice assistant in a separate thread so it doesn't freeze the camera."""
        if self.listening:
            self.status_text.set("Voice Assistant is already listening...")
            return
            
        self.listening = True
        self.status_text.set("Starting Voice Assistant...")
        self.voice_thread = threading.Thread(target=self._voice_listener_thread, daemon=True)
        self.voice_thread.start()

    def _voice_listener_thread(self) -> None:
        """
        DIANWEI: Put the Siri integration code here.
        This runs in the background. It listens to the mic and converts speech to text.
        """
        recognizer = self.sr.Recognizer()
        with self.sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening for Siri commands...")
            
            try:
                # Listens for user input
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing speech...")
                
                # Convert to text (Using Google's free API for testing)
                command = recognizer.recognize_google(audio)
                print(f"User said: {command}")
                self.status_text.set(f"Voice Command: {command}")
                
                # DIANWEI: Add your logic here to link 'command' with the object detection results.
                
            except self.sr.WaitTimeoutError:
                print("No speech detected within the timeout.")
            except self.sr.UnknownValueError:
                print("Could not understand audio.")
            except self.sr.RequestError as e:
                print(f"Could not request results; {e}")
            finally:
                self.listening = False

    def _open_camera_if_needed(self) -> bool:
        if self.camera is not None and self.camera.isOpened():
            return True

        try:
            self.camera = open_camera(
                self.cv2,
                camera_index=self.args.camera,
                width=self.args.width,
                height=self.args.height,
            )
        except RuntimeError as exc:
            self.messagebox.showerror("Camera Error", str(exc))
            self.status_text.set("Failed to open camera")
            return False
        return True

    def start_detection(self, mode: str) -> None:
        """Starts real-time detection or ROI detection."""
        if not self._open_camera_if_needed():
            return

        self.canvas.delete("placeholder")
        self.mode = mode
        self.running = True
        self.read_failures = 0
        self.frame_times.clear()
        self.previous_time = time.perf_counter()

        if mode == "realtime":
            self._clear_selection()
            self.reselect_button.configure(state=self.tk.DISABLED)
            self.status_text.set("Real-time: Detecting entire frame")
        else:
            self.reset_roi()
            self.reselect_button.configure(state=self.tk.NORMAL)

        if self.after_id is None:
            self._update_frame()

    def stop_detection(self) -> None:
        """Stops detection and releases the camera."""
        self.running = False
        self.mode = None

        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.reselect_button.configure(state=self.tk.DISABLED)
        self.status_text.set("Detection stopped")

    def close(self) -> None:
        self.stop_detection()
        self.root.destroy()

    def reset_roi(self) -> None:
        """Clears the previous ROI, allowing the user to redraw it."""
        self._clear_selection()
        if self.mode == "roi":
            self.status_text.set("Please drag to draw a box on the right screen")

    def _clear_selection(self) -> None:
        self.roi = None
        self.drag_start = None
        if self.selection_rectangle is not None:
            self.canvas.delete(self.selection_rectangle)
            self.selection_rectangle = None

    def _clamp_canvas_point(self, x: int, y: int) -> tuple[int, int]:
        x = max(0, min(self.DISPLAY_WIDTH - 1, x))
        y = max(0, min(self.DISPLAY_HEIGHT - 1, y))
        return x, y

    def _on_mouse_down(self, event: Any) -> None:
        if not self.running or self.mode != "roi":
            return

        self.roi = None
        self.drag_start = self._clamp_canvas_point(event.x, event.y)
        if self.selection_rectangle is not None:
            self.canvas.delete(self.selection_rectangle)

        x, y = self.drag_start
        self.selection_rectangle = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#fbbf24",
            width=3,
        )

    def _on_mouse_drag(self, event: Any) -> None:
        if self.drag_start is None or self.selection_rectangle is None:
            return
        x, y = self._clamp_canvas_point(event.x, event.y)
        self.canvas.coords(
            self.selection_rectangle,
            self.drag_start[0],
            self.drag_start[1],
            x,
            y,
        )

    def _on_mouse_up(self, event: Any) -> None:
        if (
            self.drag_start is None
            or self.selection_rectangle is None
            or self.latest_frame_shape is None
        ):
            return

        end_x, end_y = self._clamp_canvas_point(event.x, event.y)
        start_x, start_y = self.drag_start
        canvas_x1, canvas_x2 = sorted((start_x, end_x))
        canvas_y1, canvas_y2 = sorted((start_y, end_y))
        self.drag_start = None

        if canvas_x2 - canvas_x1 < 10 or canvas_y2 - canvas_y1 < 10:
            self.status_text.set("ROI box too small, please redraw")
            self._clear_selection()
            return

        frame_height, frame_width = self.latest_frame_shape[:2]
        x1 = int(canvas_x1 * frame_width / self.DISPLAY_WIDTH)
        x2 = int(canvas_x2 * frame_width / self.DISPLAY_WIDTH)
        y1 = int(canvas_y1 * frame_height / self.DISPLAY_HEIGHT)
        y2 = int(canvas_y2 * frame_height / self.DISPLAY_HEIGHT)
        self.roi = (x1, y1, x2, y2)

        self.canvas.itemconfigure(self.selection_rectangle, outline="#22c55e")
        self.status_text.set("ROI Detection: Detecting only inside the green box")

    def _predict(self, frame: Any) -> tuple[Any, int]:
        result = self.model.predict(source=frame, **self.predict_options)[0]
        detected_count = len(result.boxes) if result.boxes is not None else 0
        return result.plot(), detected_count

    def _calculate_fps(self) -> float:
        current_time = time.perf_counter()
        self.frame_times.append(current_time - self.previous_time)
        self.previous_time = current_time
        average_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / average_time if average_time > 0 else 0.0

    def _show_frame(self, frame: Any) -> None:
        resized = self.cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        rgb_frame = self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB)
        pil_image = self.Image.fromarray(rgb_frame)
        self.photo_image = self.ImageTk.PhotoImage(image=pil_image)
        self.canvas.itemconfigure(self.image_item, image=self.photo_image)
        self.canvas.tag_lower(self.image_item)

    def _update_frame(self) -> None:
        """Reads a frame, executes the current detection mode, and updates the interface."""
        self.after_id = None
        if not self.running or self.camera is None:
            return

        success, frame = self.camera.read()
        if not success:
            self.read_failures += 1
            if self.read_failures >= 30:
                self.messagebox.showerror("Camera Error", "Failed to read camera frame continuously.")
                self.stop_detection()
                return
            self.after_id = self.root.after(30, self._update_frame)
            return

        self.read_failures = 0
        self.latest_frame_shape = frame.shape
        annotated_frame = frame.copy()
        detected_count = 0

        try:
            if self.mode == "realtime":
                annotated_frame, detected_count = self._predict(frame)
            elif self.mode == "roi" and self.roi is not None:
                x1, y1, x2, y2 = self.roi
                roi_frame = frame[y1:y2, x1:x2]
                if roi_frame.size > 0:
                    annotated_roi, detected_count = self._predict(roi_frame)
                    annotated_frame[y1:y2, x1:x2] = annotated_roi
        except Exception as exc:
            self.messagebox.showerror("Detection Error", f"An error occurred during detection:\n{exc}")
            self.stop_detection()
            return

        fps = self._calculate_fps()
        self.cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}  Objects: {detected_count}",
            (15, 30),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            self.cv2.LINE_AA,
        )
        self._show_frame(annotated_frame)

        if self.mode == "realtime" and not self.listening:
            self.status_text.set(
                f"Real-time | Detected {detected_count} objects | {fps:.1f} FPS"
            )
        elif self.mode == "roi" and self.roi is not None and not self.listening:
            self.status_text.set(
                f"ROI Mode | Detected {detected_count} objects | {fps:.1f} FPS"
            )

        self.after_id = self.root.after(1, self._update_frame)


def main() -> int:
    args = parse_args()
    cv2, YOLO, Image, ImageTk, tk, messagebox, sr = load_dependencies()

    root = tk.Tk()
    root.withdraw()

    print(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        messagebox.showerror(
            "Model Load Failure",
            f"Failed to load model:\n{exc}\n\nPlease check network connection or model weight path.",
        )
        root.destroy()
        return 1

    ObjectDetectionApp(
        root=root,
        args=args,
        model=model,
        cv2=cv2,
        image_module=Image,
        image_tk_module=ImageTk,
        tk=tk,
        messagebox=messagebox,
        sr=sr,
    )
    root.deiconify()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
