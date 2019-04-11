from tkinter import *
from tkinter import font
from tkinter import ttk
from tkinter import filedialog

import re
from collections import deque
import json
import os.path
import platform

from functools import partial
import argparse


class Annotator(Frame):
    def __init__(self, parent, emotions, annotator_name):
        Frame.__init__(self, parent)
        self.Version = "Sentence Annotator v1"
        self.OS = platform.system().lower()
        self.parent = parent
        self.fileName = ""
        self.debug = False
        self.colorAllChunk = True
        self.recommendFlag = True
        self.history = deque(maxlen=20)
        self.currentContent = deque(maxlen=1)
        self.emotions = emotions
        self.annotator_name = annotator_name

        self.texts = None
        self.num_texts = 0
        self.current_text = None
        self.current_idx = None
        self.current_label = None


        self.buttons = None
        self.default_button_color = None
        # default GUI display parameter
        self.textRow = 24
        self.textColumn = 5

        self.keepRecommend = True
        self.selectColor = 'light salmon'
        self.textFontStyle = "Times"

        self.texts = []
        self.annotations = {}

        self.current_text = []
        self.current_annotations = []

        self.parent.title(self.Version)
        self.pack(fill=BOTH, expand=True)

        for idx in range(0,self.textColumn):
            self.columnconfigure(idx, weight =2)
        # self.columnconfigure(0, weight=2)
        self.columnconfigure(self.textColumn+2, weight=1)
        self.columnconfigure(self.textColumn+4, weight=1)
        for idx in range(0,self.textRow):
            self.rowconfigure(idx, weight =1)

        self.lbl = Label(self, text="File: no file is opened")
        self.lbl.grid(sticky=W, pady=4, padx=5)
        self.fnt = font.Font(family=self.textFontStyle,size=self.textRow,weight="bold",underline=0)
        self.text = Text(self, font=self.fnt, selectbackground=self.selectColor)
        self.text.grid(row=1, column=0, columnspan=self.textColumn, rowspan=self.textRow, padx=12, sticky=E+W+S+N)

        self.sb = Scrollbar(self)
        self.sb.grid(row = 1, column = self.textColumn, rowspan = self.textRow, padx=0, sticky = E+W+S+N)
        self.text['yscrollcommand'] = self.sb.set
        self.sb['command'] = self.text.yview
        # self.sb.pack()

        abtn = Button(self, text="Open", command=self.onOpen)
        abtn.grid(row=3, column=self.textColumn +1)

        nextButton = Button(self, text="Next", command=self.onNext)
        nextButton.grid(row=4, column=self.textColumn +1)
        self.parent.bind("<Right>", self.onNext)

        backButton = Button(self, text="Back", command=self.onBack)
        backButton.grid(row=5, column=self.textColumn +1)
        self.parent.bind("<Left>", self.onBack)

        exportbtn = Button(self, text="Export", command=self.saveAnnotations)
        exportbtn.grid(row=8, column=self.textColumn + 1)
        self.parent.bind("<Control-Key-s>", self.saveAnnotations)


        cbtn = Button(self, text="Quit", command=self.quit)
        cbtn.grid(row=9, column=self.textColumn + 1)


        self.current_label_var = StringVar()
        self.current_label_var.set("None")

        self.current_idx_var = StringVar()
        self.current_idx_var.set("0/{0}".format(self.num_texts))

        self.polarityLabel = Label(self, text="Polarity: ", textvariable=self.current_label_var, foreground="Blue", font=(self.textFontStyle, 14, "bold"))
        self.polarityLabel.grid(row=1, column=self.textColumn +1, pady=4)

        self.progress = Label(self, text="Progress: ", textvariable=self.current_idx_var, foreground="Blue", font=(self.textFontStyle, 14, "bold"))
        self.progress.grid(row=2, column=self.textColumn +1, pady=4)

        self.setButtons()


    def onOpen(self):
        ftypes = [('all files', '.*'), ('text files', '.txt'), ('ann files', '.ann')]
        dlg = filedialog.Open(self, filetypes = ftypes)
        # file_opt = options =  {}
        # options['filetypes'] = [('all files', '.*'), ('text files', '.txt')]
        # dlg = tkFileDialog.askopenfilename(**options)
        fl = dlg.show()
        if fl != '':
            self.text.delete("1.0",END)
            self.texts = self.readFile(fl)
            self.num_texts = len(self.texts)
            self.current_text = "\n-----------------------------------\n".join(self.texts[0].split("\t"))
            self.current_idx = 0

            self.current_idx_var.set("{0}/{1}".format(self.current_idx+1, self.num_texts))

            self.progress = Label(self, text="Progress: ", textvariable=self.current_idx_var, foreground="Blue", font=(self.textFontStyle, 14, "bold"))
            self.progress.grid(row=2, column=self.textColumn +1, pady=4)



            #print(self.current_text)


            self.text.insert(END, self.current_text)
            self.setNameLabel("File: " + fl)

        # If you already have annotations for this instance, set current
        # annotations to these previous annotations
        try:
            self.current_annotations = self.annotations[self.current_idx]
        except KeyError:
            self.current_annotations = {"anger": { '0': None, '1': None, '2': None, '3': None},
                                        "disgust": { '0': None, '1': None, '2': None, '3': None},
                                        "fear": { '0': None, '1': None, '2': None, '3': None},
                                        "happiness": { '0': None, '1': None, '2': None, '3': None},
                                        "sadness": { '0': None, '1': None, '2': None, '3': None},
                                        "surprise": { '0': None, '1': None, '2': None, '3': None}
                                        }

    def onNext(self, event=None):

        # Save the annotations before moving to next instance
        self.annotations[self.current_idx] = self.current_annotations

        # Set current_idx to next idx
        self.current_idx += 1

        # When you come to the end of the annotations, loop around
        if self.current_idx == self.num_texts:
            self.current_idx = 0

        # If you already have annotations for this instance, set current
        # annotations to these previous annotations
        try:
            self.current_annotations = self.annotations[self.current_idx]
        except KeyError:
            self.current_annotations = {"anger": { '0': None, '1': None, '2': None, '3': None},
                                        "disgust": { '0': None, '1': None, '2': None, '3': None},
                                        "fear": { '0': None, '1': None, '2': None, '3': None},
                                        "happiness": { '0': None, '1': None, '2': None, '3': None},
                                        "sadness": { '0': None, '1': None, '2': None, '3': None},
                                        "surprise": { '0': None, '1': None, '2': None, '3': None}
                                        }

        # Color buttons red if their label is already annotated
        for label in self.buttons.keys():
            emo, num = label.split("-")
            if self.current_annotations[emo][num] == "most":
                self.pressed(label)
            elif self.current_annotations[emo][num] == "least":
                self.repressed(label)
            else:
                self.unpressed(label)

        # For debugging, print current annotations
        print(self.current_annotations)

        # For debugging, print if first or last instance
        if self.current_idx == 0:
            print("First instance")
        elif self.current_idx == len(self.texts) - 1:
            print("Last instance")

        # Get the new label and text and update the widgets parameters
        # to show these
        self.current_text = "\n-----------------------------------\n".join(self.texts[self.current_idx].split("\t"))
        self.text.delete("1.0",END)
        self.text.insert(END, self.current_text)

        self.current_idx_var.set("{0}/{1}".format(self.current_idx+1, self.num_texts))

        self.progress = Label(self, text="Progress: ", textvariable=self.current_idx_var, foreground="Blue", font=(self.textFontStyle, 14, "bold"))
        self.progress.grid(row=2, column=self.textColumn +1, pady=4)

    def onBack(self, event=None):

        # Save current annotations
        self.annotations[self.current_idx] = self.current_annotations

        # Set current idx to previous idx
        self.current_idx -= 1

        # If you go to the end of the dataset, loop back around and start
        # with the last instance
        if self.current_idx < 0:
            self.current_idx = self.num_texts - 1

        # If you already have annotations for this instance, set current
        # annotations to these previous annotations
        try:
            self.current_annotations = self.annotations[self.current_idx]
        except KeyError:
            self.current_annotations = {"anger": { '0': None, '1': None, '2': None, '3': None},
                                        "disgust": { '0': None, '1': None, '2': None, '3': None},
                                        "fear": { '0': None, '1': None, '2': None, '3': None},
                                        "happiness": { '0': None, '1': None, '2': None, '3': None},
                                        "sadness": { '0': None, '1': None, '2': None, '3': None},
                                        "surprise": { '0': None, '1': None, '2': None, '3': None}
                                        }
        print(self.current_annotations)


        # Color buttons red if their label is already annotated
        for label in self.buttons.keys():
            emo, num = label.split("-")
            if self.current_annotations[emo][num] == "most":
                self.pressed(label)
            elif self.current_annotations[emo][num] == "least":
                self.repressed(label)
            else:
                self.unpressed(label)

        # For debugging, print current annotations
        print(self.current_annotations)

        # For debugging, print if first or last instance
        if self.current_idx == 0:
            print("First instance")
        elif self.current_idx == len(self.texts) - 1:
            print("Last instance")

        # Get the new label and text and update the widgets parameters
        # to show these
        self.current_text = "\n-----------------------------------\n".join(self.texts[self.current_idx].split("\t"))
        self.text.delete("1.0",END)
        self.text.insert(END, self.current_text)

        self.current_idx_var.set("{0}/{1}".format(self.current_idx+1, self.num_texts))

        self.progress = Label(self, text="Progress: ", textvariable=self.current_idx_var, foreground="Blue", font=(self.textFontStyle, 14, "bold"))
        self.progress.grid(row=2, column=self.textColumn +1, pady=4)

    def readFile(self, filename):
        to_annotate = open(filename, encoding="utf8").readlines()
        self.fileName = filename

        basename = os.path.basename(filename)
        ann_file = os.path.join("anns", basename + "." + self.annotator_name + ".ann")

        if os.path.exists(ann_file):
            print("Using previous annotations")
            with open(ann_file, "r") as infile:
                annotations = json.load(infile)

            #loading a json loads everything as a string, which doesn't work with the next and
            #back button setup, so we have to convert the outermost keys to integers

            anns = {}
            for key, value in annotations.items():
                anns[int(key)] = value
            self.annotations = anns


            #If it exists, set current annotations to first annotations, otherwise initialize
            try:
                self.current_annotations = self.annotations[0]
            except KeyError:
                self.current_annotations = {"anger": { '0': None, '1': None, '2': None, '3': None},
                                        "disgust": { '0': None, '1': None, '2': None, '3': None},
                                        "fear": { '0': None, '1': None, '2': None, '3': None},
                                        "happiness": { '0': None, '1': None, '2': None, '3': None},
                                        "sadness": { '0': None, '1': None, '2': None, '3': None},
                                        "surprise": { '0': None, '1': None, '2': None, '3': None}
                                        }

            # Color buttons red if their label is already annotated
            for label in self.buttons.keys():
                emo, num = label.split("-")
                if self.current_annotations[emo][num] == "most":
                    self.pressed(label)
                elif self.current_annotations[emo][num] == "least":
                    self.repressed(label)
                else:
                    self.unpressed(label)

        else:
            print("No annotations found")

        return to_annotate

    def setFont(self, value):
        _family=self.textFontStyle
        _size = value
        _weight="bold"
        _underline=0
        fnt = font.Font(family= _family,size= _size,weight= _weight,underline= _underline)
        Text(self, font=fnt)

    def setNameLabel(self, new_file):
        self.lbl.config(text=new_file)


    def checkLabel(self, event=None, label=None):
        emo, num = label.split("-")
        if self.current_annotations[emo][num] == None:
            self.setmostLabel(label)
        elif self.current_annotations[emo][num] == "most":
            self.setleastLabel(label)
        else:
            self.removeLabel(label)

    def setmostLabel(self, label):
        print("set {} to most".format(label))
        self.pressed(label)
        emo, num = label.split("-")
        self.current_annotations[emo][num] = "most"


    def setleastLabel(self, label):
        print("set {} to least".format(label))
        self.repressed(label)
        emo, num = label.split("-")
        self.current_annotations[emo][num] = "least"

    def removeLabel(self, label):
        print("removed {}".format(label))
        self.unpressed(label)
        emo, num = label.split("-")
        self.current_annotations[emo][num] = None


    def pressed(self, label):
        self.buttons[label].configure(bg="red")

    def repressed(self, label):
        self.buttons[label].configure(bg="blue")

    def unpressed(self, label):
        self.buttons[label].configure(bg=self.default_button_color)

    ## show shortcut map
    def setButtons(self):

        # destroy all previous widgets before switching shortcut maps
        if self.buttons is not None and type(self.buttons) is type([]):
            for x in self.buttons:
                x.destroy()

        self.buttons = {}

        for i, emo in enumerate(self.emotions):
            for ann in range(4):
                nextButton = Button(self, text=emo, command=partial(self.checkLabel, label=emo + "-" + str(ann)))
                nextButton.grid(row= 4+ann, column=self.textColumn+2+i)
                self.buttons[emo + "-"+ str(ann)] = nextButton
                self.parent.bind(emo, partial(self.checkLabel, label=emo + "-" + str(ann)))

            if i == 0:
                self.default_button_color = nextButton.cget("background")


    def saveAnnotations(self, event=None):
        if (".ann" not in self.fileName) and (".txt" not in self.fileName):
            out_error = "Export only works on filename ended in .ann or .txt!\nPlease rename file."

        # make sure current annotations are included
        self.annotations[self.current_idx] = self.current_annotations

        basename = os.path.basename(self.fileName)
        fileName = "anns/" + basename + "." + self.annotator_name + ".ann"
        print("Saving annotations to " + fileName)

        with open(fileName, 'w') as out:
            json.dump(self.annotations, out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotions", nargs="+", default=["anger", "disgust", "fear", "happiness", "sadness", "surprise"])
    parser.add_argument("--annotator_name", default="annotator1")

    args = parser.parse_args()

    print("SUTDAnnotator launched!")
    print(("OS:%s")%(platform.system()))
    root = Tk()
    root.geometry("1700x700+200+200")
    app = Annotator(root, args.emotions, args.annotator_name)
    app.setFont(17)

if __name__ == "__main__":

    main()


