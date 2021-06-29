import PIL
from PIL import Image,ImageTk
from tkinter import *
import tkinter as tk
import time
from environment.variable import CONFIG_PATH
import pickle
import ui as Ui
from imutils import face_utils
import cv2
import numpy as np
import dlib
import os
from environment.variable import ROOT_DIR,DATASET_DIR
from multiprocessing import Process,Queue
import uuid


class Recorder:
    def __init__(self):
        self.init_window()
        self.load_configuration()
        self.init_variables()
        self.init_components()
        self.init_camera()
        self.run()

    def init_window(self):
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.configuration['Camera Width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.configuration['Camera Height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.configuration['Frames In One Sample'])
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def load_configuration(self):
        self.configuration = pickle.load(open(CONFIG_PATH, "rb"))

    def init_variables(self):
        self.stopwatch = self.configuration['Frames In One Sample']

        self.name_dataset = self.configuration['Name Dataset']
        self.name_class = self.configuration['Name Class']

        self.switch = False
        self.tick = time.time()
        self.framerate_arr = []
        self.framecount = 0
        self.framerate_text = tk.StringVar()
        self.info_text = tk.StringVar()

        self.recorded_frames=[]

        self.face_shape =(120,120)

        self.num_workers = 3

        self.workers = []

        for i in range(self.num_workers):
            self.workers.append(Sampler(self.face_shape,self.name_dataset,self.name_class,
                                          self.configuration['Frames In One Sample']))

        self.count = 0

    def init_components(self):
        self.framerate_label = Label(self.window,textvariable=self.framerate_text)
        self.framerate_label.pack()

        self.frame = Label(self.window)
        self.frame.pack()

        self.info_label = Label(self.window,textvariable=self.info_text)
        self.info_label.pack()



        self.ui_start = Ui.LABEL_WITH_BUTTON(self.window,
                                                  '',
                                                  '[Record Start]',
                                                  self.ui_start_command)
        self.ui_start.pack()


    def event_close(self):
        for i in range(self.num_workers):
            self.workers[i].off()
        self.frame.after_cancel(self.frame_after_id)
        self.cap.release()

        cv2.destroyAllWindows()
        self.window.destroy()


        Ui.Main()

        del self

    def show_image(self,image):
        frame = cv2.flip(image, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.frame.imgtk = imgtk
        self.frame.configure(image=imgtk)

    def record(self):
        ret, frame = self.cap.read()
        if ret:
            now = time.time()
            self.framerate_arr.append(1/(now - self.tick))
            if len(self.framerate_arr) > 30:
                self.framerate_arr.pop(0)
                self.framerate_text.set("FPS: "+ str( int( sum(self.framerate_arr[15:])/15)))

            self.tick = now
            if self.framecount % self.configuration['Sampling Rate'] == 0:
                self.show_image(frame)

            if self.switch:

                self.info_text.set("Recording: " + str(self.stopwatch-1))

                self.recorded_frames.append(frame)
                self.stopwatch = self.stopwatch - 1
                if self.stopwatch == 0:

                    index = self.count % self.num_workers
                    self.workers[index].queue.put((self.recorded_frames,index,self.count))

                    self.count = self.count + 1
                    self.recorded_frames = []
                    self.switch = False
                    self.stopwatch = self.configuration['Frames In One Sample']

            else:
                self.info_text.set("Press Record To Start Recording")
            self.frame_after_id = self.frame.after(1, self.record)
            self.framecount = self.framecount + 1

    def ui_start_command(self):
        self.switch = True

    def run(self):
        self.record()
        self.window.mainloop()

class Sampler():
    def __init__(self,face_shape,name_dataset,name_class,fps):
        self.name_dataset = name_dataset
        self.name_class = name_class

        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(os.path.join(ROOT_DIR, "environment", "shape_predictor_68_face_landmarks.dat"))

        self.count = 0
        self.face_shape = face_shape
        self.fps = fps

        self.switch = True

        self.init_directories()

        self.queue = Queue()
        self.process = Process(target=self.main)
        self.process.start()

    def init_directories(self):

        if not os.path.exists(os.path.join(DATASET_DIR,self.name_dataset,'face_image',self.name_class)):
            os.makedirs(os.path.join(DATASET_DIR,self.name_dataset,'face_image',self.name_class))

        if not os.path.exists(os.path.join(DATASET_DIR,self.name_dataset,'face_video',self.name_class)):
            os.makedirs(os.path.join(DATASET_DIR,self.name_dataset,'face_video',self.name_class))

        if not os.path.exists(os.path.join(DATASET_DIR,self.name_dataset,'original_video',self.name_class)):
            os.makedirs(os.path.join(DATASET_DIR,self.name_dataset,'original_video',self.name_class))

    def off(self):
        self.switch = False
        self.process.kill()

    def main(self):
        while True:
            if not self.queue.empty():
                task = self.queue.get()
                self.data = task[0]
                self.worker_name = task[1]
                self.filename = str(task[2])+'_' + str(uuid.uuid4())
                self.pipeline()
                self.count = self.count + 1



                print('[Worker ',self.worker_name,']','[',self.count,'/',self.queue.qsize()+self.count ,'] saved.' , ' [Total: ',task[2]+1,' done]')

            else:
                if self.switch == False:
                    return

    def pipeline(self):


        self.face_cropper()
        if self.data is None:
            print('[Worker ', self.worker_name, ']', 'No Face Detected Error.', ' Killing Task.')
            return

        self.video_saver('original_video')
        self.image_saver('face_image')
        self.video_saver('face_video')


    def image_saver(self,definition):


        path = os.path.join(DATASET_DIR,self.name_dataset,definition,self.name_class,self.filename)
        if not os.path.exists(path):
            os.makedirs(path)

        for i,image in enumerate(self.face):
            cv2.imwrite(os.path.join(path,'frame_'+ str(i).zfill(5)+'.png' ),image)


    def video_saver(self,definition):

        random_filename = self.filename+ '.mp4'
        path = os.path.join(DATASET_DIR,self.name_dataset,definition,self.name_class,random_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if definition == 'face_video':
            out = cv2.VideoWriter(path, fourcc, self.fps, (self.face[0].shape[1], self.face[0].shape[0]))
            for frame in self.face:
                out.write(frame)
            out.release()
        elif definition == 'original_video':
            out = cv2.VideoWriter(path, fourcc, self.fps, (self.data[0].shape[1], self.data[0].shape[0]))
            for frame in self.data:
                out.write(frame)
            out.release()

    def face_cropper(self):
        if self.data is None:
            return

        if len(self.data) != self.fps:
            self.data = None
            return

        outputs = []
        for image in self.data:

            rects = self.face_detector(image, 0)
            if len(rects) != 1:
                self.data = None
                return

            shape = self.face_predictor(image, rects[0])
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
            ratio = 70 / w

            image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
            x = x * ratio
            y = y * ratio
            w = w * ratio
            h = h * ratio
            midy = y + h / 2
            midx = x + w / 2
            xshape = self.face_shape[1] / 2
            yshape = self.face_shape[0] / 2

            cropped = image[int(midy - yshape):int(midy + yshape), int(midx - xshape):int(midx + xshape)]

            outputs.append(cropped)

        self.face = outputs
