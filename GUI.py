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
        canvas = tkinter.Canvas(frame, height=300, width=300)
        image = Image.open(filepath)
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.grid(row=1, column=0)

def show_answer():
    global filepath
    lbl.config(text=run(filepath))


root = tkinter.Tk()
root.title("CNN")
root.geometry("500x300")

frame = tkinter.Frame(
    root,
    padx=10,
    pady=10
)
frame.pack(anchor="nw")

bttn = tkinter.Button(frame, text="Запустить", command=show_answer)
bttn.grid(row=0, column=1)

open_button = ttk.Button(frame, text="Открыть файл", command=open_file)
open_button.grid(column=0, row=0)

lbl = tkinter.Label(frame, text='')
lbl.grid(row=1, column=1)


root.mainloop()