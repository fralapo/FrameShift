import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import threading
import os
from pathlib import Path
import shutil
import sys
import io
import logging
import cv2
import math
import platform
import subprocess

# --- UI Constants ---
ALL_POSSIBLE_OBJECT_LABELS = sorted(list(set(["face", "person"] + [
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
])))

DEFAULT_WEIGHTS = {"face": 1.0, "person": 0.8, "default": 0.5}

# Attempt to import frameshift modules
try:
    from frameshift.main import process_video, mux_video_audio_with_ffmpeg, get_cv2_interpolation_flag
    from frameshift.weights_parser import parse_object_weights
    from frameshift.utils.detection import Detector
    FRAMESHIFT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import frameshift modules. Error: {e}")
    FRAMESHIFT_AVAILABLE = False
    # Fallback definitions
    def process_video(*args, **kwargs):
        messagebox.showerror("Error", "Module 'frameshift.main.process_video' not found.")
        return None
    def mux_video_audio_with_ffmpeg(*args, **kwargs): return False
    def parse_object_weights(value):
        if not value: return {}
        try: return dict(item.split(':') for item in value.split(','))
        except Exception:
            messagebox.showerror("Error", "Invalid object weights format.")
            return {}
    def get_cv2_interpolation_flag(name): return 0
    class Detector:
        def __init__(self): print("Warning: Using placeholder Detector class.")

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tooltip_window, text=self.text, background="#FFFFE0", relief="solid", borderwidth=1, wraplength=300)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

class FrameShiftGUI:
    class TqdmRedirector(io.TextIOBase):
        def __init__(self, widget_update_callback):
            self.widget_update_callback = widget_update_callback
        def write(self, text):
            if text.strip(): self.widget_update_callback(text.strip())
            return len(text)
        def flush(self): pass

    def __init__(self, master):
        self.master = master
        master.title("FrameShift GUI")
        master.geometry("850x950")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.is_batch = tk.BooleanVar(value=False)
        self.aspect_ratio_selection = tk.StringVar(value="9:16 (Vertical)")
        self.custom_aspect_ratio = tk.StringVar(value="4:3")
        self.output_height = tk.StringVar() 
        self.interpolation_method = tk.StringVar(value="linear")
        self.enable_padding = tk.BooleanVar(value=True)
        self.padding_type = tk.StringVar(value="black")
        self.blur_amount = tk.IntVar(value=5)
        self.padding_color_value = tk.StringVar(value="(0, 0, 0)")
        self.content_opacity = tk.DoubleVar(value=1.0)
        self.progress_status_text = tk.StringVar(value="")
        self.save_log_file = tk.BooleanVar(value=False)
        self.log_file_path = tk.StringVar()
        self.active_weights_dict = DEFAULT_WEIGHTS.copy()
        self.weight_vars = {}
        self.input_label_text = tk.StringVar()
        self.output_label_text = tk.StringVar()
        self.video_info_text = tk.StringVar()
        self.cancel_event = threading.Event()
        
        self.STANDARD_RESOLUTIONS = {
            '9:16': ['1920', '1280', '1080', '720'],
            '1:1': ['1080', '768'],
            '4:5': ['1350', '1080'],
            '16:9': ['1080', '720', '480'],
            'custom': ['1080', '1920', '720', '1280']
        }

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Accent.TButton", font=('calibri', 10, 'bold'), foreground='green')
        style.configure("Cancel.TButton", font=('calibri', 10, 'bold'), foreground='red')
        style.configure("Italic.TLabel", font=('calibri', 9, 'italic'))

        menubar = tk.Menu(master)
        master.config(menu=menubar)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about_dialog)
        help_menu.add_command(label="Quick Guide", command=self.show_quick_guide)
        menubar.add_cascade(label="Help", menu=help_menu)

        # Scrollable container: the settings don't always fit the window
        # (small screens, or many object-detection rows), so the whole content
        # area lives inside a Canvas that can scroll vertically.
        outer_frame = ttk.Frame(master)
        outer_frame.pack(fill=tk.BOTH, expand=True)
        self.main_canvas = tk.Canvas(outer_frame, highlightthickness=0)
        vscrollbar = ttk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=vscrollbar.set)
        vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main_frame = ttk.Frame(self.main_canvas, padding="10")
        self._main_window_id = self.main_canvas.create_window((0, 0), window=main_frame, anchor="nw")

        def _on_main_frame_configure(event):
            # Keep the scrollable region matched to the content height.
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        main_frame.bind("<Configure>", _on_main_frame_configure)

        def _on_main_canvas_configure(event):
            # Stretch the inner frame to the canvas width so widgets fill it.
            self.main_canvas.itemconfigure(self._main_window_id, width=event.width)
        self.main_canvas.bind("<Configure>", _on_main_canvas_configure)

        def _on_mousewheel(event):
            if event.num == 4:        # Linux scroll up
                self.main_canvas.yview_scroll(-1, "units")
            elif event.num == 5:      # Linux scroll down
                self.main_canvas.yview_scroll(1, "units")
            else:                     # Windows / macOS
                self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.main_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.main_canvas.bind_all("<Button-5>", _on_mousewheel)

        io_frame = ttk.LabelFrame(main_frame, text="Input & Output", padding="10")
        io_frame.pack(fill=tk.X, expand=False, pady=5)
        io_frame.columnconfigure(1, weight=1)
        ttk.Label(io_frame, textvariable=self.input_label_text).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.input_entry = ttk.Entry(io_frame, textvariable=self.input_path, state="readonly")
        self.input_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.browse_input_btn = ttk.Button(io_frame, text="Browse...", command=self.browse_input)
        self.browse_input_btn.grid(row=0, column=2, padx=5, pady=5)
        self.resolution_label = ttk.Label(io_frame, textvariable=self.video_info_text)
        self.resolution_label.grid(row=1, column=1, padx=5, pady=(0,5), sticky=tk.W)
        ttk.Label(io_frame, textvariable=self.output_label_text).grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_entry = ttk.Entry(io_frame, textvariable=self.output_path, state="readonly")
        self.output_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.browse_output_btn = ttk.Button(io_frame, text="Browse...", command=self.browse_output)
        self.browse_output_btn.grid(row=2, column=2, padx=5, pady=5)
        self.batch_check = ttk.Checkbutton(io_frame, text="Batch Process", variable=self.is_batch, command=self.toggle_batch_mode)
        self.batch_check.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        settings_outer_frame = ttk.Frame(main_frame)
        settings_outer_frame.pack(fill=tk.X, expand=False, pady=5)
        settings_outer_frame.columnconfigure(0, weight=1)
        settings_outer_frame.columnconfigure(1, weight=1)
        
        main_settings_frame = ttk.LabelFrame(settings_outer_frame, text="Main Settings", padding="10")
        main_settings_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        ttk.Label(main_settings_frame, text="Aspect Ratio:").pack(anchor=tk.W, padx=5, pady=(5,0))
        ratio_options = ["9:16 (Vertical)", "1:1 (Square)", "4:5 (Social)", "16:9 (Horizontal)", "Custom..."]
        self.ratio_combo = ttk.Combobox(main_settings_frame, textvariable=self.aspect_ratio_selection, values=ratio_options, state="readonly")
        self.ratio_combo.pack(fill=tk.X, padx=5, pady=(0,5))
        self.ratio_combo.bind("<<ComboboxSelected>>", self.on_ratio_select)
        self.custom_ratio_entry = ttk.Entry(main_settings_frame, textvariable=self.custom_aspect_ratio)
        ttk.Label(main_settings_frame, text="Output Height (pixels):").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.height_combo = ttk.Combobox(main_settings_frame, textvariable=self.output_height, state="readonly")
        self.height_combo.pack(fill=tk.X, padx=5, pady=(0,5))

        padding_settings_frame = ttk.LabelFrame(settings_outer_frame, text="Padding Settings", padding="10")
        padding_settings_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        self.padding_checkbox = ttk.Checkbutton(padding_settings_frame, text="Enable Padding", variable=self.enable_padding, command=self.toggle_padding_options)
        self.padding_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        self.padding_options_frame = ttk.Frame(padding_settings_frame)
        self.padding_options_frame.pack(fill=tk.X, expand=True)
        ttk.Label(self.padding_options_frame, text="Padding Type:").pack(anchor=tk.W, padx=5, pady=(5,0))
        padding_types = ["black", "blur", "color"]
        self.padding_type_combo = ttk.Combobox(self.padding_options_frame, textvariable=self.padding_type, values=padding_types, state="readonly")
        self.padding_type_combo.pack(fill=tk.X, padx=5, pady=(0,5))
        self.padding_type_combo.bind("<<ComboboxSelected>>", self.toggle_padding_details)
        self.blur_options_frame = ttk.Frame(self.padding_options_frame)
        ttk.Label(self.blur_options_frame, text="Blur Amount (0-10):").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.blur_slider = ttk.Scale(self.blur_options_frame, from_=0, to=10, variable=self.blur_amount, orient=tk.HORIZONTAL, command=lambda s: self.blur_amount.set(int(float(s))))
        self.blur_slider.pack(fill=tk.X, padx=5, pady=(0,5))
        self.color_options_frame = ttk.Frame(self.padding_options_frame)
        ttk.Label(self.color_options_frame, text="Padding Color:").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.color_entry = ttk.Entry(self.color_options_frame, textvariable=self.padding_color_value)
        self.color_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=(0,5))
        self.color_button = ttk.Button(self.color_options_frame, text="Choose...", command=self.choose_padding_color)
        self.color_button.pack(side=tk.LEFT, padx=(0,5), pady=(0,5))

        visuals_frame = ttk.LabelFrame(main_frame, text="Visual Quality", padding="10")
        visuals_frame.pack(fill=tk.X, expand=False, pady=5)
        visuals_frame.columnconfigure(1, weight=1)
        ttk.Label(visuals_frame, text="Interpolation:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        interpolation_options = ["lanczos", "cubic", "linear", "area", "nearest"]
        self.interpolation_combo = ttk.Combobox(visuals_frame, textvariable=self.interpolation_method, values=interpolation_options, state="readonly")
        self.interpolation_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(visuals_frame, text="Content Opacity (0.0-1.0):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.opacity_slider = ttk.Scale(visuals_frame, from_=0.0, to=1.0, variable=self.content_opacity, orient=tk.HORIZONTAL)
        self.opacity_slider.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        advanced_frame = ttk.LabelFrame(main_frame, text="Object Detection Weights", padding="10")
        advanced_frame.pack(fill=tk.X, expand=False, pady=5)
        info_label = ttk.Label(advanced_frame, text="Note: Object detection (excl. 'face') is only enabled for objects with a weight > 0.", style="Italic.TLabel")
        info_label.pack(fill=tk.X, pady=(0, 5))
        self.weights_list_frame = ttk.Frame(advanced_frame)
        self.weights_list_frame.pack(fill=tk.X, expand=True, pady=5)
        add_weight_frame = ttk.Frame(advanced_frame)
        add_weight_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Label(add_weight_frame, text="Add Object:").pack(side=tk.LEFT, padx=(0, 5))
        self.add_weight_combo = ttk.Combobox(add_weight_frame, state="readonly", values=[])
        self.add_weight_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.add_weight_btn = ttk.Button(add_weight_frame, text="Add", command=self.add_weight)
        self.add_weight_btn.pack(side=tk.LEFT, padx=5)

        action_frame = ttk.LabelFrame(main_frame, text="Actions & Status", padding="10")
        action_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        action_frame.columnconfigure(0, weight=1)
        action_frame.rowconfigure(3, weight=1)
        log_check_frame = ttk.Frame(action_frame)
        log_check_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5)
        ttk.Checkbutton(log_check_frame, text="Save detailed log to file", variable=self.save_log_file).pack(side=tk.LEFT)
        buttons_frame = ttk.Frame(action_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        self.start_button = ttk.Button(buttons_frame, text="START PROCESS", command=self.start_processing_thread, style="Accent.TButton")
        self.start_button.pack()
        self.cancel_button = ttk.Button(buttons_frame, text="CANCEL PROCESS", command=self.cancel_processing, style="Cancel.TButton")
        self.progress_bar = ttk.Progressbar(action_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.grid(row=1, column=2, padx=5, pady=5, sticky=tk.EW)
        action_frame.columnconfigure(2, weight=1)
        self.progress_status_label = ttk.Label(action_frame, textvariable=self.progress_status_text, anchor=tk.W)
        self.progress_status_label.grid(row=2, column=0, columnspan=4, padx=5, pady=(0,5), sticky=tk.EW)
        self.log_text = tk.Text(action_frame, height=10, state="disabled", wrap=tk.WORD, background="#F0F0F0")
        log_scrollbar = ttk.Scrollbar(action_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.NSEW)
        log_scrollbar.grid(row=3, column=3, sticky=tk.NS)

        self.toggle_batch_mode() 
        self.update_output_height_options()
        self.on_ratio_select()
        self.toggle_padding_options()
        self._render_weights_ui()
        self._apply_tooltips()
        self._initialize_detector()
        
    def _initialize_detector(self):
        self.detector = None
        if FRAMESHIFT_AVAILABLE:
            try:
                self.log_message("Initializing Detector...")
                self.detector = Detector()
                self.log_message("Detector initialized successfully.")
            except Exception as e:
                self.log_message(f"Error initializing Detector: {e}", "ERROR")
                messagebox.showerror("Detector Error", f"Could not initialize the detector: {e}\nFrameShift may not work correctly.")
    
    def _apply_tooltips(self):
        ToolTip(self.input_entry, "Path to the input video file or folder.")
        ToolTip(self.output_entry, "Path to the output video file or folder.")
        ToolTip(self.batch_check, "Check to process all videos in the input folder and save them to the output folder.")
        ToolTip(self.ratio_combo, "Choose a standard aspect ratio or select 'Custom' to enter one manually.")
        ToolTip(self.custom_ratio_entry, "Enter a custom aspect ratio. Examples: '4:3' or a number like '1.33'.")
        ToolTip(self.height_combo, "Choose a standard height for the output video.")
        ToolTip(self.interpolation_combo, "Algorithm for resizing frames. 'lanczos'/'cubic' for quality, 'area' for downscaling, 'linear' for speed.")
        ToolTip(self.opacity_slider, "Opacity of the main video content. If < 1.0, the content is blended with a blurred version of the full original frame.")
        ToolTip(self.padding_checkbox, "If checked, fits the content within the frame by adding bars (letterboxing/pillarboxing).")
        ToolTip(self.padding_type_combo, "Choose the type of bars to add: 'blur', 'black', or 'color'.")
        ToolTip(self.add_weight_combo, "Choose an object to add to the detection weights list.")
        ToolTip(self.start_button, "Start the reframing process with the current settings.")
        ToolTip(self.cancel_button, "Request to stop the current process. Should take effect relatively quickly.")

    def _render_weights_ui(self):
        for widget in self.weights_list_frame.winfo_children(): widget.destroy()
        self.weight_vars.clear()
        
        current_weights = self.active_weights_dict
        
        current_labels = current_weights.keys()
        available_to_add = sorted([label for label in ALL_POSSIBLE_OBJECT_LABELS if label not in current_labels])
        self.add_weight_combo['values'] = available_to_add
        if available_to_add: self.add_weight_combo.set(available_to_add[0])
        else: self.add_weight_combo.set("")

        sorted_labels = sorted(current_weights.keys(), key=lambda x: (x == 'default', x))
        for label in sorted_labels:
            value = current_weights[label]
            row_frame = ttk.Frame(self.weights_list_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=f"{label}:", width=15).pack(side=tk.LEFT, padx=5)
            weight_var = tk.DoubleVar(value=value)
            self.weight_vars[label] = weight_var
            slider = ttk.Scale(row_frame, from_=0, to=1.0, variable=weight_var, orient=tk.HORIZONTAL)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            value_label = ttk.Label(row_frame, width=5)
            value_label.pack(side=tk.LEFT, padx=5)
            def update_value_label(val, lbl=value_label): lbl.config(text=f"{float(val):.2f}")
            slider.config(command=update_value_label)
            update_value_label(value)
            if label != "default":
                remove_btn = ttk.Button(row_frame, text="X", width=3, command=lambda l=label: self.remove_weight(l))
                remove_btn.pack(side=tk.LEFT, padx=5)

    def add_weight(self):
        label_to_add = self.add_weight_combo.get()
        if not label_to_add: return
        if label_to_add not in self.active_weights_dict:
            self.active_weights_dict[label_to_add] = 0.5
            self._render_weights_ui()
        else:
            messagebox.showwarning("Object Exists", f"The object '{label_to_add}' is already in the list.")

    def remove_weight(self, label_to_remove):
        if label_to_remove in self.active_weights_dict and label_to_remove != "default":
            del self.active_weights_dict[label_to_remove]
            self._render_weights_ui()
            
    def get_current_weights_string(self):
        return ",".join([f"{label}:{var.get():.2f}" for label, var in self.weight_vars.items()])

    def show_about_dialog(self):
        messagebox.showinfo("About FrameShift GUI", "FrameShift GUI\n\nA graphical user interface for the FrameShift tool.")

    def show_quick_guide(self):
        messagebox.showinfo("Quick Guide", "1. Select input/output.\n2. Adjust settings.\n3. Click 'START PROCESS'.")

    def on_ratio_select(self, event=None):
        if self.aspect_ratio_selection.get() == "Custom...":
            self.custom_ratio_entry.pack(fill=tk.X, padx=5, pady=(0,5), after=self.ratio_combo)
        else:
            self.custom_ratio_entry.pack_forget()
        self.update_output_height_options()

    def update_output_height_options(self):
        selection = self.aspect_ratio_selection.get()
        if selection == "Custom...":
            ratio_key = 'custom'
        else:
            ratio_key = selection.split(" ")[0]
        heights = self.STANDARD_RESOLUTIONS.get(ratio_key, self.STANDARD_RESOLUTIONS['custom'])
        self.height_combo['values'] = heights
        if heights:
            self.output_height.set(heights[0])

    def update_progress_label(self, text):
        self.master.after(0, lambda: self.progress_status_text.set(text))

    def log_message(self, message, level="INFO"):
        def _update_log():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, f"[{level}] {message}\n")
            self.log_text.config(state="disabled")
            self.log_text.see(tk.END)
        self.master.after(0, _update_log)
        if level in ("ERROR", "WARNING"): print(f"[{level}] {message}")

    def browse_input(self):
        self.video_info_text.set("")
        path = filedialog.askdirectory(title="Select Folder") if self.is_batch.get() else filedialog.askopenfilename(title="Select Video", filetypes=(("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")))
        if path:
            self.input_path.set(path)
            self.log_message(f"Input: {path}")
            if not self.is_batch.get() and os.path.isfile(path):
                try:
                    cap = cv2.VideoCapture(path)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        info_str = f"Resolution: {width}x{height}"
                        if height > 0:
                            common_divisor = math.gcd(width, height)
                            ar_w = width // common_divisor
                            ar_h = height // common_divisor
                            info_str += f" (Aspect Ratio: {ar_w}:{ar_h})"
                        self.video_info_text.set(info_str)
                        cap.release()
                except Exception as e:
                    self.log_message(f"Could not read video resolution: {e}", "WARNING")
                if not self.output_path.get():
                    p = Path(path)
                    self.output_path.set(str(p.parent / f"{p.stem}_reframed{p.suffix}"))
                    self.log_message(f"Suggested output: {self.output_path.get()}")

    def browse_output(self):
        if self.is_batch.get(): path = filedialog.askdirectory(title="Select Output Folder")
        else:
            p = Path(self.input_path.get()) if self.input_path.get() and os.path.isfile(self.input_path.get()) else None
            default_name = f"{p.stem}_reframed{p.suffix}" if p else ""
            path = filedialog.asksaveasfilename(title="Save Video As...", defaultextension=".mp4", initialfile=default_name, filetypes=(("MP4 Files", "*.mp4"), ("All files", "*.*")))
        if path:
            self.output_path.set(path)
            self.log_message(f"Output: {path}")

    def toggle_batch_mode(self):
        self.video_info_text.set("")
        if self.is_batch.get():
            self.input_label_text.set("Input Folder:")
            self.output_label_text.set("Output Folder:")
        else:
            self.input_label_text.set("Input Video:")
            self.output_label_text.set("Output Video:")
        self.log_message(f"Batch mode {'enabled' if self.is_batch.get() else 'disabled'}. Paths have been reset.")
        self.input_path.set("")
        self.output_path.set("")

    def toggle_padding_options(self):
        state = 'normal' if self.enable_padding.get() else 'disabled'
        for child in self.padding_options_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for sub_child in child.winfo_children():
                     try: sub_child.configure(state=state)
                     except tk.TclError: pass
            else:
                 try: child.configure(state=state)
                 except tk.TclError: pass
        if self.enable_padding.get(): self.toggle_padding_details()
        else:
            self.blur_options_frame.pack_forget()
            self.color_options_frame.pack_forget()

    def toggle_padding_details(self, event=None):
        if not self.enable_padding.get(): return
        selected = self.padding_type.get()
        self.blur_options_frame.pack_forget()
        self.color_options_frame.pack_forget()
        if selected == "blur": self.blur_options_frame.pack(fill=tk.X, expand=True, pady=(5,0))
        elif selected == "color": self.color_options_frame.pack(fill=tk.X, expand=True, pady=(5,0))

    def choose_padding_color(self):
        color_code = colorchooser.askcolor(title="Choose Color")
        if color_code and color_code[0]:
            rgb_tuple_str = f"({int(color_code[0][0])},{int(color_code[0][1])},{int(color_code[0][2])})"
            self.padding_color_value.set(rgb_tuple_str)
            self.log_message(f"Padding color selected: RGB {rgb_tuple_str}")

    def get_current_aspect_ratio(self):
        selection = self.aspect_ratio_selection.get()
        return self.custom_aspect_ratio.get() if selection == "Custom..." else selection.split(" ")[0]

    def _validate_inputs(self):
        if not self.input_path.get() or not self.output_path.get():
            messagebox.showerror("Error", "Please select both an input and an output.")
            return False
        if not self.output_height.get():
            messagebox.showerror("Error", "Please select an output height.")
            return False
        if self.save_log_file.get():
            log_path = filedialog.asksaveasfilename(title="Save Log File As...", defaultextension=".log", filetypes=(("Log files", "*.log"), ("Text files", "*.txt")))
            self.log_file_path.set(log_path)
            if not log_path:
                messagebox.showwarning("Log Canceled", "Log file saving was canceled. The log will only be shown on screen.")
                self.save_log_file.set(False)
        return True

    def start_processing_thread(self):
        if not FRAMESHIFT_AVAILABLE or not self.detector:
            messagebox.showerror("Error", "FrameShift modules are not available or initialized.")
            return
        if not self._validate_inputs(): return

        self.cancel_event.clear()
        self.start_button.pack_forget()
        self.cancel_button.pack()
        self.progress_bar.start(10)
        self.log_message("Starting process...")
        threading.Thread(target=self._process_videos_in_thread, daemon=True).start()

    def cancel_processing(self):
        self.log_message("Cancellation requested...", "WARNING")
        self.cancel_event.set()
        self.cancel_button.config(state="disabled", text="CANCELLING...")

    def _process_videos_in_thread(self):
        redirector = self.TqdmRedirector(self.update_progress_label)
        original_stderr = sys.stderr
        sys.stderr = redirector
        
        try:
            common_args = {
                "ratio": self.get_current_aspect_ratio(), "apply_padding_flag": self.enable_padding.get(),
                "padding_type_str": self.padding_type.get(), "padding_color_str": self.padding_color_value.get(),
                "blur_amount_param": self.blur_amount.get(), "output_target_height": int(self.output_height.get()),
                "interpolation_flag": get_cv2_interpolation_flag(self.interpolation_method.get()),
                "content_opacity": self.content_opacity.get(),
                "object_weights_map": parse_object_weights(self.get_current_weights_string()),
                "detector": self.detector,
                "cancel_event": self.cancel_event, # Pass the cancel event
            }
            ffmpeg_path = shutil.which("ffmpeg")
            
            if self.is_batch.get():
                in_dir = Path(self.input_path.get())
                out_dir = Path(self.output_path.get())
                videos_to_process = [p for p in in_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
                num_videos = len(videos_to_process)
                self.master.after(0, lambda: self.progress_bar.config(mode='determinate', maximum=num_videos, value=0))

                for i, vid_path in enumerate(videos_to_process):
                    if self.cancel_event.is_set():
                        self.log_message("Batch process canceled by user.", "WARNING")
                        break
                    self.update_progress_label("")
                    self.log_message(f"Processing file {i+1}/{num_videos}: {vid_path.name}")
                    final_output_file = out_dir / f"{vid_path.stem}_reframed{vid_path.suffix}"

                    temp_video_file_or_status = process_video(
                        input_path=str(vid_path),
                        output_path=str(final_output_file), # Explicitly pass output_path
                        **common_args # common_args should not contain output_path
                    )

                    if self.cancel_event.is_set(): # Check immediately after process_video
                        self.log_message(f"Cancellation detected after processing {vid_path.name}.", "WARNING")
                        # temp_video_file_or_status would be "cancelled" or "cancelled_error_cleanup"
                        # The file is already handled by process_video if cancelled.
                        break # Break from batch loop

                    self._handle_muxing_and_temp_file(str(vid_path), temp_video_file_or_status, str(final_output_file), ffmpeg_path)
                    self.master.after(0, lambda v=i+1: self.progress_bar.config(value=v))
            else: # Single file processing
                self.update_progress_label("")
                self.master.after(0, lambda: self.progress_bar.config(mode='indeterminate'))

                temp_video_file_or_status = process_video(
                    input_path=self.input_path.get(),
                    output_path=self.output_path.get(), # Explicitly pass output_path
                    **common_args # common_args should not contain output_path
                )

                if self.cancel_event.is_set(): # Check immediately after process_video
                    self.log_message("Cancellation detected during single video processing.", "WARNING")
                    # temp_video_file_or_status would be "cancelled" or "cancelled_error_cleanup"
                    # The file is already handled by process_video. No further action on file needed here.
                else:
                    # Only mux if not cancelled.
                    self._handle_muxing_and_temp_file(self.input_path.get(), temp_video_file_or_status, self.output_path.get(), ffmpeg_path)
            
            if not self.cancel_event.is_set():
                self.log_message("Processing completed.")
            else:
                self.log_message("Processing was cancelled by the user.", "WARNING")

        except Exception as e:
            self.log_message(f"CRITICAL ERROR: {e}", "ERROR")
            import traceback
            self.log_message(traceback.format_exc(), "ERROR")
            error_message = str(e)
            self.master.after(0, lambda msg=error_message: messagebox.showerror("Error", f"An unexpected error occurred: {msg}"))
        finally:
            sys.stderr = original_stderr
            self.master.after(0, self._processing_finished)

    def _handle_muxing_and_temp_file(self, original_input, temp_video_or_status, final_output, ffmpeg_path):
        if self.cancel_event.is_set():
            self.log_message(f"Muxing skipped for {Path(original_input).name} due to cancellation.", "WARNING")
            # temp_video_or_status should be 'cancelled' or 'cancelled_error_cleanup', file already handled by process_video
            return

        # Check if process_video returned a status string indicating cancellation or error
        if isinstance(temp_video_or_status, str) and temp_video_or_status.startswith("cancelled"):
            self.log_message(f"Muxing skipped for {Path(original_input).name} as processing was cancelled.", "INFO")
            # The file should have been cleaned up by process_video already.
            return

        temp_video_path = temp_video_or_status # It's a path string if not cancelled

        if not temp_video_path or not os.path.exists(temp_video_path):
            self.log_message(f"Processing failed or was cancelled for {Path(original_input).name}, no temporary file available for muxing.", "ERROR")
            return

        if not ffmpeg_path:
            self.log_message(f"FFmpeg not found, saving video without audio.", "WARNING")
            try: shutil.move(temp_video_path, final_output) # Use temp_video_path
            except Exception as e: self.log_message(f"Failed to move file: {e}", "ERROR")
            return
        
        self.log_message(f"Muxing audio for {Path(final_output).name}...")
        success = mux_video_audio_with_ffmpeg(original_input, temp_video_path, final_output, ffmpeg_path) # Use temp_video_path
        if success:
            self.log_message(f"Audio muxed successfully. Final file: {final_output}")
            try: os.remove(temp_video_path) # Use temp_video_path
            except OSError as e: self.log_message(f"Could not remove temp file: {e}", "WARNING")
        else:
            self.log_message(f"Audio muxing failed. Moving video without audio.", "WARNING")
            try: shutil.move(temp_video_path, final_output) # Use temp_video_path
            except Exception as e: self.log_message(f"Failed to move temp file: {e}", "ERROR")

    def _processing_finished(self):
        self.progress_bar.stop()
        self.progress_bar.config(value=0)
        status = "Operation canceled." if self.cancel_event.is_set() else "Operation finished."
        self.update_progress_label(status)
        self.cancel_button.pack_forget()
        self.cancel_button.config(state="normal", text="CANCEL PROCESS")
        self.start_button.pack()
        self.log_message(f"--- {status} ---", "INFO")
        if not self.cancel_event.is_set():
            messagebox.showinfo("Completed", "Processing has finished.")
            # Open output location if processing was successful
            if self.output_path.get():
                self.log_message(f"Attempting to open output location: {self.output_path.get()}")
                self._open_output_location(self.output_path.get())
            else:
                self.log_message("Output path is not set, cannot open location.", "WARNING")

    def _open_output_location(self, path_str: str):
        try:
            path = Path(path_str)
            system = platform.system()

            # Determine the actual path to open (directory or file's parent directory)
            if path.is_file():
                open_path = str(path.parent) # Open parent directory for a file
                select_path = str(path)      # Path to the file itself for selection (if supported)
            elif path.is_dir():
                open_path = str(path)        # Open the directory itself
                select_path = None           # No specific file to select
            else:
                self.log_message(f"Output path {path_str} does not exist or is not a file/directory. Cannot open.", "ERROR")
                return

            self.log_message(f"System: {system}, Path to open/reveal: {open_path if not select_path else select_path}")

            if system == "Windows":
                if select_path and Path(select_path).exists(): # Check if select_path is valid
                    # Using /select, will open the folder and select the file
                    subprocess.run(['explorer', '/select,', select_path], check=True)
                elif Path(open_path).exists(): # Fallback to opening the directory
                    subprocess.run(['explorer', open_path], check=True)
                else:
                    self.log_message(f"Path {open_path} or {select_path} not found on Windows.", "ERROR")
            elif system == "Darwin": # macOS
                if select_path and Path(select_path).exists():
                     # -R reveals the file in Finder
                    subprocess.run(['open', '-R', select_path], check=True)
                elif Path(open_path).exists():
                    subprocess.run(['open', open_path], check=True) # Open the directory
                else:
                    self.log_message(f"Path {open_path} or {select_path} not found on macOS.", "ERROR")
            elif system == "Linux":
                if Path(open_path).exists(): # xdg-open usually opens the directory
                    subprocess.run(['xdg-open', open_path], check=True)
                else:
                    self.log_message(f"Path {open_path} not found on Linux.", "ERROR")
            else:
                self.log_message(f"Unsupported operating system: {system}. Cannot open output location automatically.", "WARNING")
        except FileNotFoundError:
            self.log_message(f"File explorer command not found. Ensure it's in your PATH.", "ERROR")
        except subprocess.CalledProcessError as e:
            self.log_message(f"Error opening output location: {e}", "ERROR")
        except Exception as e:
            self.log_message(f"An unexpected error occurred while opening output location: {e}", "ERROR")

def main_gui():
    root = tk.Tk()
    if not FRAMESHIFT_AVAILABLE:
        messagebox.showwarning("Dependencies Missing", "FrameShift modules not found. The app will run in limited mode.")
    app = FrameShiftGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main_gui()
