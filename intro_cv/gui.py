from tkinter import *
def click_button(context):
    context.configure(text = "I got clicked")
root = Tk()   ##intialize the root class
root.title("Photobooth App")   ##title the window
root.geometry('700x200') ##set the widthxheight of the app
a = Label(root, text ="Click Me")  #add text to the main window
# a.pack()  #pack the text onto the app, push it to the first available spot
a.grid() #put the text onto the first available grid
btn = Button(root, text = "Click me" ,
             fg = "red", command=lambda: click_button(a))
btn.grid(column=1, row=0)
root.mainloop() 