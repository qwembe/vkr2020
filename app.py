import tkinter as tk  # python 3
from tkinter import *
from tkinter import messagebox
from tkinter import font  as tkfont  # python 3

import modules.entry_module as entry

from settings.opt import *

import logging

log = logging.getLogger(__name__)


class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.dataset = tk.IntVar(value=1)
        log.debug(str(self.dataset.get()))
        self.detector = tk.IntVar(0)
        self.matcher = tk.IntVar(0)
        self.other = tk.IntVar(0)
        self.enable_gt = tk.BooleanVar(0)
        self.enable_knn = tk.BooleanVar(0)
        self.disable_vis = tk.BooleanVar(0)

        self.frames = {}
        for F in (Main, PageOne, Settings, Dataset, Detector, Matcher, Other):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("Main")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.update()
        frame.tkraise()

    def run(self):
        self.mainloop()

    def update(self):
        pass


class Main(Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = LabelFrame(self, text="Visual Odometry")
        button1 = Button(label, text="Start", command=lambda: self.scenario1())
        button2 = Button(label, text="Settings", command=lambda: controller.show_frame("Settings"))
        label.pack(side="top", fill="x", pady=10)
        label_frame = tk.LabelFrame(self, text="Current settings")

        button1.pack()
        button2.pack()
        label_frame.pack()

        label_ds = tk.LabelFrame(label_frame, text="Dataset")
        self.ds_info = tk.Label(label_ds)
        label_dr = tk.LabelFrame(label_frame, text="Detector")
        self.dr_info = tk.Label(label_dr)
        label_mt = tk.LabelFrame(label_frame, text="Matcher")
        self.mt_info = tk.Label(label_mt)
        # label_ot = tk.LabelFrame(label_frame, text="Other")
        # self.ot_info = tk.Label(label_ot)

        label_ds.pack()
        label_dr.pack()
        label_mt.pack()
        # label_ot.pack()
        self.ds_info.pack()
        self.dr_info.pack()
        self.mt_info.pack()
        # self.ot_info.pack()

    def update(self):
        # log.debug(str(self.controller.dataset.get()))

        # For debug
        # self.controller.detector = tk.IntVar(value=1)
        log.debug(str(self.controller.dataset.get()))
        #

        self.ds_info.configure(text=DATASET[self.controller.dataset.get()])
        self.dr_info.configure(text=DETECTORS[self.controller.detector.get()])
        self.mt_info.configure(text=MATCHERS[self.controller.matcher.get()])
        # self.ot_info.configure(text=OTHER[self.controller.other.get()])

    def scenario1(self):
        print("Scenario ")
        contr_val = self.controller.matcher.get()
        if self.controller.detector.get() == 0 and (contr_val == 0 or contr_val == 1):
            messagebox.showerror("Error", "Fast algorithm does'n have descriptors. Choose other configuration")
            return
        res = entry.run(dataset=DATASET[self.controller.dataset.get()],
                        detetector=DETECTORS[self.controller.detector.get()],
                        matcher=MATCHERS[self.controller.matcher.get()],
                        other=[self.controller.enable_gt.get(),
                               self.controller.enable_knn.get(),
                               self.controller.disable_vis.get()])
        average_ate, final_stat, fps, peak = res
        messagebox.showinfo("Stats", "Programm ended with: \n" + "average ATE: " + str(average_ate)
                            + "\nfinal ATE: " + str(final_stat)
                            + "\nfps: " + str(fps)
                            + "\nmemory peak: " + str(peak))
        print("Scenario  - ended with " + str(res))


class PageOne(Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="You not supposed to be here!")
        label.pack()


class Settings(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Settings", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        b1 = tk.Button(self, text="Dataset", command=lambda: controller.show_frame("Dataset"))
        b2 = tk.Button(self, text="Detectors", command=lambda: controller.show_frame("Detector"))
        b3 = tk.Button(self, text="Matchers", command=lambda: controller.show_frame("Matcher"))
        # b4 = tk.Button(self, text="Other", command=lambda: controller.show_frame("Other"))
        b5 = tk.Button(self, text="Back", command=lambda: controller.show_frame("Main"))

        b1.pack()
        b2.pack()
        b3.pack()
        # b4.pack()
        b5.pack()


class Dataset(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Dataset", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        r1 = Radiobutton(self, text="Kitti", variable=controller.dataset, value=0)
        r2 = Radiobutton(self, text="TUM", variable=controller.dataset, value=1)
        # r3 = Radiobutton(self, text="None", variable=controller.dataset, value=2)
        bb = Button(self, text="Back", command=lambda: controller.show_frame("Settings"))

        r1.pack()
        r1.select()
        r2.pack()
        # r3.pack()
        bb.pack()


class Detector(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Detector", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        r1 = Radiobutton(self, text="FAST", variable=controller.detector, value=0)
        r2 = Radiobutton(self, text="SIFT", variable=controller.detector, value=1)
        r3 = Radiobutton(self, text="ORB", variable=controller.detector, value=2)
        bb = Button(self, text="Back", command=lambda: controller.show_frame("Settings"))

        r1.pack()
        r1.select()
        r2.pack()
        r3.pack()
        bb.pack()


# calcOpticalFlowFarneback  ---!
class Matcher(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Matcher", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        r1 = Radiobutton(self, text="Bruteforce", variable=controller.matcher, value=0)
        r2 = Radiobutton(self, text="FLANN", variable=controller.matcher, value=1)
        r3 = Radiobutton(self, text="Lucas-Kanade", variable=controller.matcher, value=2)
        bb = Button(self, text="Back", command=lambda: controller.show_frame("Settings"))

        r1.pack()
        r1.select()
        r2.pack()
        r3.pack()
        bb.pack()


# Not used correctly! Not usable
class Other(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Other", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        r1 = Radiobutton(self, text="None", variable=controller.other, value=0)
        r2 = Radiobutton(self, text="Kaplassian filter", variable=controller.other, value=1)
        r3 = Radiobutton(self, text="Bundle Adjustsment", variable=controller.other, value=2)
        c1 = Checkbutton(text="Enable GT", variable=controller.enable_gt, onvalue=True, offvalue=False)
        c2 = Checkbutton(text="Enable knnMatch", variable=controller.enable_knn, onvalue=True, offvalue=False)
        c3 = Checkbutton(text="Disable visualisation", variable=controller.disable_vis, onvalue=True, offvalue=False)
        bb = Button(self, text="Back", command=lambda: controller.show_frame("Settings"))

        # r1.pack()
        # r1.select()
        # r2.pack()
        # r3.pack()
        c1.pack()
        c2.pack()
        c3.pack()
        bb.pack()
