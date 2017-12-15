import tkinter as tk
from tkinter import messagebox as mbox
import NeuralNetwork as NN

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__()
        master.minsize(width=400, height=350)
        master.title("Neural Network Optimizer")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.txtInput  = tk.Text(self, height=1, width=10)
        self.txtOutput = tk.Text(self, height=1, width=10)
        self.btnInput  = tk.Button(self, text="Generate", command=self.generate)

        self.txtInput.grid(row=1, column=1, pady=(10,10))
        self.txtOutput.grid(row=2, column=1, pady=(10,10))
        self.btnInput.grid(row=3, column=1, pady=(10,10))

    def generate(self):
        try:
            inBits  = int(self.txtInput.get("1.0", "end-1c"))
            outBits = int(self.txtOutput.get("1.0", "end-1c"))
            print("Given Config: {0} {1}".format(inBits, outBits))

            # generate multiple neural networks and find optimal structure
            for i in range(3):
                for j in range():

            nn = NN.NeuralNetwork(inBits, [3,2], outBits, 0.3)
            nn.print_structure()
        except:
            mbox.showerror("Error", "Input and Output must be numbers!")
        pass


root = tk.Tk()
app = App(master=root)
app.mainloop()

