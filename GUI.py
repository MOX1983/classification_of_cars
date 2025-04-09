import tkinter
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from Test import run


photo = None
filepath = None

def open_file():
    global photo
    global filepath
    filepath = filedialog.askopenfilename()
    if filepath != "":
        canvas = tkinter.Canvas(root, height=300, width=300)
        image = Image.open(filepath)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.grid(row=1, column=0)

def show_answer():
    global filepath
    lbl.config(text=run(filepath))


root = tkinter.Tk()
root.title("car or truck")
root.geometry("500x300")

bttn = tkinter.Button(text="Запустить", command=show_answer)
bttn.grid(row=0, column=1)

open_button = ttk.Button(text="Открыть файл", command=open_file)
open_button.grid(column=0, row=0)

lbl = tkinter.Label(text='')
lbl.grid(row=1, column=1)


root.mainloop()