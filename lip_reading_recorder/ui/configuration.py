from tkinter import *
import os
from environment.variable import CONFIG_PATH,DATASET_DIR,MODEL_LIST
from tkinter import ttk
import pickle
import shutil
import ui as Ui

class Configuration:
    def __init__(self):

        self.init_window()
        self.load_configuration()
        self.init_components()
        self.run()

    def load_configuration(self):
        self.configuration = pickle.load(open(CONFIG_PATH, "rb"))


    def init_window(self):
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_components(self):

        self.notebook = ttk.Notebook(self.window)
        self.init_record_tab()
        self.notebook.grid(row=0)

        self.ui_save = Ui.LABEL_WITH_BUTTON(self.window,
                                            '',
                                            '[SAVE]',
                                            self.ui_save_command)
        self.ui_save.frame.grid(row=1, padx=20, pady=5)


    def init_record_tab(self):
        index_row = 0
        self.record_setting_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.record_setting_tab, text='Record')
        ############################
        self.ui_name_dataset = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Name Dataset',
                                                    str(self.configuration['Name Dataset']),
                                                    20)
        self.ui_name_dataset.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_name_class = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Name Class',
                                                    str(self.configuration['Name Class']),
                                                    20)
        self.ui_name_class.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_number_of_workers = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Number Of Workers',
                                                    str(self.configuration['Number Of Workers']),
                                                    20)
        self.ui_number_of_workers.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_sampling_rate = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Sampling Rate',
                                                    str(self.configuration['Sampling Rate']),
                                                    20)
        self.ui_sampling_rate.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_camera_width = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                   'Camera Width',
                                                   str(self.configuration['Camera Width']),
                                                   20)
        self.ui_camera_width.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_camera_height = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Camera Height',
                                                    str(self.configuration['Camera Height']),
                                                    20)
        self.ui_camera_height.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_frames_in_one_sample = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                           'Frames In One Sample',
                                                           str(self.configuration['Frames In One Sample']),
                                                           20)
        self.ui_frames_in_one_sample.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_delete_all_dataset = Ui.LABEL_WITH_BUTTON(self.record_setting_tab,
                                                          'DELETE ALL DATASET',
                                                          'DELETE',
                                                          self.ui_delete_all_dataset_command)
        self.ui_delete_all_dataset.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1
        ############################
        self.ui_rename_dataset_naturally = Ui.LABEL_WITH_BUTTON(self.record_setting_tab,
                                                          'RENAME DATASET NATURALLY',
                                                          'RENAME',
                                                          self.ui_rename_dataset_naturally_command)
        self.ui_rename_dataset_naturally.frame.grid(row=index_row, padx=20, pady=5)
        index_row += 1

    def ui_rename_dataset_naturally_command(self):
        path = os.path.join(DATASET_DIR,str(self.ui_name_dataset.entry.get()))
        if os.path.exists(path):
            for definition in  os.listdir(path):
                definitioin_path = os.path.join(path,definition)
                for classes in os.listdir(definitioin_path):
                    class_path = os.path.join(definitioin_path,classes)
                    tmp_count = 0
                    for each in os.listdir(class_path):
                        each_path = os.path.join(class_path,each)

                        if each_path.endswith('.mp4'):
                            print(each_path, ' ==> ', 'data' + str(tmp_count)+'.mp4' )
                            os.rename(each_path, os.path.join(class_path,'data_'+ str(tmp_count)+'.mp4' ) )
                        else:
                            print(each_path, ' ==> ', 'data' + str(tmp_count))
                            os.rename(each_path, os.path.join(class_path, 'data_' + str(tmp_count)))
                        tmp_count +=1
                        #print(each_path)


        else:
            print("DATASET NOT EXIST")


    def ui_delete_all_dataset_command(self):
        if os.path.exists(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)

    def ui_save_command(self):

        config = {'Name Dataset' : str(self.ui_name_dataset.entry.get()),
                  'Name Class' : str(self.ui_name_class.entry.get()),
                  'Sampling Rate': int(self.ui_sampling_rate.entry.get()),
                  'Camera Width': int(self.ui_camera_width.entry.get()),
                  'Camera Height': int(self.ui_camera_height.entry.get()),
                  'Frames In One Sample': int(self.ui_frames_in_one_sample.entry.get()),
                  'Number Of Workers': int(self.ui_number_of_workers.entry.get()),
                  }

        f = open(CONFIG_PATH, "wb")
        pickle.dump(config, f)
        f.close()

        self.event_close()


    def event_close(self):
        self.window.destroy()
        Ui.Main()
        del self

    def run(self):
        self.window.mainloop()

