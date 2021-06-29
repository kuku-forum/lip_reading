from tkinter import *
import os
from environment.variable import CONFIG_PATH
import pickle
import ui as Ui

class Main:
    def __init__(self):
        self.create_config_file()
        self.init_window()
        self.init_components()
        self.run()

    def init_window(self):
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def create_config_file(self):
        if not os.path.exists(CONFIG_PATH):
            config ={'Sampling Rate':4,
                     'Camera Width':1280 ,
                     'Camera Height':720,
                     'Frames In One Sample':29,
                     'Name Dataset':'dataset1',
                     'Number Of Workers': '2',
                     'Name Class':'alfa'}
            f = open(CONFIG_PATH, "wb")
            pickle.dump(config, f)
            f.close()

    def init_components(self):

        self.ui_recorder = Ui.LABEL_WITH_BUTTON(self.window,
                                                  '',
                                                  '[Record]',
                                                  self.ui_recorder_command)
        self.ui_recorder.frame.grid(row=0,padx=70,pady=10)


        self.ui_configuration = Ui.LABEL_WITH_BUTTON(self.window,
                                                  '',
                                                  '[Configuration]',
                                                  self.ui_configuration_command)
        self.ui_configuration.frame.grid(row=2,padx=70,pady=10)



    def event_close(self):
        self.window.destroy()
        self.window.quit()
        del self


    def ui_configuration_command(self):
        self.window.destroy()
        Ui.Configuration()
        del self

    def ui_recorder_command(self):
        self.window.destroy()
        Ui.Recorder()
        del self

    def run(self):
        self.window.mainloop()