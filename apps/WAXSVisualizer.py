import tkinter as tk
from tkinter import filedialog, Frame
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm
import math

class ImageApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Easy GIWAXS")
        
        # cut  img
        self.img=None
        self.yp=None
        self.xp=None
        self.cut_img=None
        
        # Left frame to contain buttons and inputs
        self.left_frame = Frame(self.root)
        self.left_frame.pack(side=tk.TOP, padx=2, pady=20)

        # Coordinate labels and input boxes inside the left frame
        self.create_coordinate_input()

        # Create a button to load the image
        self.load_btn = tk.Button(self.left_frame, text="Load tif", command=self.load_image)
        self.load_btn.grid(row=2, column=0, pady=5, columnspan=2)

        # Create a Frame for the matplotlib canvas (Image)
        self.canvas_frame = Frame(self.root)
        self.canvas_frame.pack(side=tk.LEFT, pady=20, fill=tk.BOTH, expand=True)

        # Create a Frame for the matplotlib canvas (Integration)
        self.canvas_frame2 = Frame(self.root)
        self.canvas_frame2.pack(side=tk.RIGHT, pady=20, fill=tk.BOTH, expand=True)

        # display q
        self.r_label = tk.Label(self.left_frame, text="|q| =")
        self.r_label.grid(row=3, column=0, pady=5, sticky=tk.W)

        self.r_value = tk.StringVar()  # This will store the computed value
        self.r_display = tk.Label(self.left_frame, textvariable=self.r_value)
        self.r_display.grid(row=3, column=1, pady=5, sticky=tk.W)

        # Create a button to reset the program
        self.reset_btn = tk.Button(self.left_frame, text="Reset tif", command=self.reset_program)
        self.reset_btn.grid(row=2, column=2, pady=5, columnspan=2)

        self.select_rect_btn = tk.Button(self.left_frame, text="Select Int Region", command=self.enable_select_rect_mode)
        self.select_rect_btn.grid(row=4, column=0, pady=5, columnspan=2)

        self.reset_int_btn = tk.Button(self.left_frame, text="Reset Int", command=self.reset_int)
        self.reset_int_btn.grid(row=4, column=2, pady=5, columnspan=2)

        self.canvas=None
        self.canvas2=None


    def create_coordinate_input(self):
        self.label_x_start = tk.Label(self.left_frame, text="qxy min (1/A):")
        self.label_x_start.grid(row=0, column=0, pady=5, sticky=tk.W)
        self.entry_x_start = tk.Entry(self.left_frame)
        self.entry_x_start.grid(row=0, column=1, pady=5)

        self.label_x_end = tk.Label(self.left_frame, text="qxy max (1/A):")
        self.label_x_end.grid(row=0, column=2, pady=5, sticky=tk.W)
        self.entry_x_end = tk.Entry(self.left_frame)
        self.entry_x_end.grid(row=0, column=3, pady=5)

        self.label_y_start = tk.Label(self.left_frame, text="qz min (1/A):")
        self.label_y_start.grid(row=1, column=0, pady=5, sticky=tk.W)
        self.entry_y_start = tk.Entry(self.left_frame)
        self.entry_y_start.grid(row=1, column=1, pady=5)

        self.label_y_end = tk.Label(self.left_frame, text="qz max (1/A):")
        self.label_y_end.grid(row=1, column=2, pady=5, sticky=tk.W)
        self.entry_y_end = tk.Entry(self.left_frame)
        self.entry_y_end.grid(row=1, column=3, pady=5)

    def get_coordinates_from_entries(self):
        return float(self.entry_x_start.get()), float(self.entry_x_end.get()), float(self.entry_y_start.get()), float(self.entry_y_end.get())

    def load_image(self):
        img_path = filedialog.askopenfilename(title="Select a .tiff image", filetypes=[("TIFF files", "*.tiff")])
        if not img_path:
            return

        self.img = Image.open(img_path)
        self.display_img_with_coordinates()

    def on_click(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            r = np.sqrt(x**2 + y**2)
            # Update the StringVar with the new value
            self.r_value.set(f"{r:.2f}")

    def display_img_with_coordinates(self):
        # Get the coordinates from the Entry widgets
        a, b, c, d = self.get_coordinates_from_entries()

        # Convert PIL image to numpy array
        img_array = np.array(self.img)
        self.yp,self.xp=img_array.shape
        lb = np.nanpercentile(img_array, 10)
        ub = np.nanpercentile(img_array, 99)

        # Create a new figure and axis to display the image
        fig, ax = plt.subplots()
        ax.imshow(img_array,interpolation='nearest', cmap=cm.jet,origin='lower',
               vmax=ub, vmin=lb,extent=[a,b,c,d])
        ax.set_xlabel('q$_{xy}$ (1/$\AA$)')
        ax.set_ylabel('q$_{z}$ (1/$\AA$)')
        
        # Embed the matplotlib figure into the tkinter window
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()

        toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.draw()

    def enable_select_rect_mode(self):
        if self.canvas:  # Ensure the canvas exists before trying to bind events
            self.cid_rect=self.canvas.mpl_connect('button_press_event', self.rect_on_press)
            self.cid_rect1=self.canvas.mpl_connect('button_release_event', self.rect_on_release)
        self.rect = None
        self.x0 = None
        self.y0 = None

    def rect_on_press(self, event):
        if event.inaxes is None: return  # Ensure the click is within the image region
        self.x0, self.y0 = event.xdata, event.ydata
        self.rect = plt.Rectangle((self.x0, self.y0), 1, 1, fill=False, edgecolor='red', linewidth=1.5)
        event.inaxes.add_patch(self.rect)

    def rect_on_release(self, event):
        if event.inaxes is None or self.rect is None: return
        self.x1, self.y1 = event.xdata, event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))

        self.rect_int()
        self.canvas.mpl_disconnect(self.cid_rect)
        self.canvas.mpl_disconnect(self.cid_rect1)

    def rect_int(self):
        # Get the coordinates from the Entry widgets
        a, b, c, d = self.get_coordinates_from_entries()
        a1= min(self.x0,self.x1)
        b1= max(self.x0,self.x1)
        c1= min(self.y0,self.y1)
        d1= max(self.y0,self.y1)

        xL=math.floor((b1-a1 )/(b-a)*self.xp)
        yL=math.floor((d1-c1)/(d-c)*self.yp)
        xs=math.floor((a1 -a)/(b-a)*self.xp)
        ys=math.floor((c1-c)/(d-c)*self.yp)
        # Convert PIL image to numpy array
        img_array = np.array(self.img)
        self.cut_img=img_array[ys:ys+yL,xs:xs+xL]

        lb = np.nanpercentile(img_array, 10)
        ub = np.nanpercentile(img_array, 99)

        # Create a new figure and axis to display the image
        fig1, ax1 = plt.subplots(2,2,figsize=(5, 3))
        plt.tight_layout()
        ax1[0,0].imshow(self.cut_img,interpolation='nearest', cmap=cm.jet,origin='lower',
               vmax=ub, vmin=lb,extent=[a1,b1,c1,d1])
        xint=np.sum(self.cut_img, axis=0)
        ax1[0,1].plot(np.linspace(a1,b1,xint.size),xint)
        ax1[0,1].set_xlabel('q$_{xy}$ (1/$\AA$)')
        ax1[0,1].set_ylabel('Intensity (a.u.)')
        yint=np.sum(self.cut_img, axis=1)
        ax1[1,0].plot(np.linspace(c1,d1,yint.size),yint)
        ax1[1,0].set_xlabel('q$_{z}$ (1/$\AA$)')
        ax1[1,0].set_ylabel('Intensity (a.u.)')
        # Embed the matplotlib figure into the tkinter window
        self.canvas2 = FigureCanvasTkAgg(fig1, master=self.canvas_frame2)
        canvas_widget2 = self.canvas2.get_tk_widget()
        canvas_widget2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas2.draw()
    
    def reset_program(self):
        # Clear the existing image canvas
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Clear the existing coordinate input values
        self.entry_x_start.delete(0, tk.END)
        self.entry_x_end.delete(0, tk.END)
        self.entry_y_start.delete(0, tk.END)
        self.entry_y_end.delete(0, tk.END)

        # Clear the sqrt(x^2 + y^2) display
        self.r_value.set("")
    
    def reset_int(self):
        # Clear the existing image canvas
        for widget in self.canvas_frame2.winfo_children():
            widget.destroy()
    

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

