import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GraphFrame(tk.Frame):
    def __init__(self, master, title, frequency, amplitude, phase):
        super().__init__(master)
        self.title = title
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot()

    def update_plot(self):
        x = np.linspace(0, 2 * np.pi, 1000)
        if self.title == "Sine Wave":
            y = self.amplitude * np.sin(self.frequency * x + self.phase)
        elif self.title == "Tangent Wave":
            y = self.amplitude * np.tan(self.frequency * x + self.phase)
        elif self.title == "Cosine Wave":
            y = self.amplitude * np.cos(self.frequency * x + self.phase)

        self.ax.clear()
        self.ax.plot(x, y, color='blue')
        self.ax.set_title(self.title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.canvas.draw()

class GraphPage(tk.Frame):
    def __init__(self, master, frequency, amplitude, phase):
        super().__init__(master)

        self.sine_frame = GraphFrame(self, "Sine Wave", frequency, amplitude, phase)
        self.sine_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.tangent_frame = GraphFrame(self, "Tangent Wave", frequency, amplitude, phase)
        self.tangent_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.cosine_frame = GraphFrame(self, "Cosine Wave", frequency, amplitude, phase)
        self.cosine_frame.pack(side=tk.LEFT, padx=10, pady=10)

class InputPage(tk.Frame):
    def __init__(self, master, switch_page_callback):
        super().__init__(master)
        self.switch_page_callback = switch_page_callback

        ttk.Label(self, text="Frequency").grid(row=0, column=0, padx=5, pady=5)
        self.frequency_entry = ttk.Entry(self)
        self.frequency_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self, text="Amplitude").grid(row=1, column=0, padx=5, pady=5)
        self.amplitude_entry = ttk.Entry(self)
        self.amplitude_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self, text="Phase").grid(row=2, column=0, padx=5, pady=5)
        self.phase_entry = ttk.Entry(self)
        self.phase_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(self, text="Confirm", command=self.confirm_input).grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def confirm_input(self):
        frequency = float(self.frequency_entry.get())
        amplitude = float(self.amplitude_entry.get())
        phase = float(self.phase_entry.get())
        self.switch_page_callback(frequency, amplitude, phase)

class SineWavePlotterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wave Plotter")
        self.geometry("1000x600")

        self.input_page = InputPage(self, self.switch_to_graph_page)
        self.graph_pages = []

        # Create 4 separate output pages
        for _ in range(4):
            graph_page = GraphPage(self, 0, 0, 0)  # Initial values are placeholders
            self.graph_pages.append(graph_page)

        self.current_page = None
        self.switch_to_input_page()

        self.create_navigation_menu()

    def create_navigation_menu(self):
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        graph_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Graph Pages", menu=graph_menu)

        for i, _ in enumerate(self.graph_pages):
            graph_menu.add_command(label=f"Graph Page {i+1}", command=lambda idx=i: self.switch_to_graph_page_index(idx))

    def switch_to_graph_page(self, frequency, amplitude, phase):
        for graph_page in self.graph_pages:
            graph_page.pack_forget()

        for i in range(4):
            graph_page = self.graph_pages[i]
            graph_page.sine_frame.frequency = frequency
            graph_page.sine_frame.amplitude = amplitude
            graph_page.sine_frame.phase = phase
            graph_page.sine_frame.update_plot()

            graph_page.tangent_frame.frequency = frequency
            graph_page.tangent_frame.amplitude = amplitude
            graph_page.tangent_frame.phase = phase
            graph_page.tangent_frame.update_plot()

            graph_page.cosine_frame.frequency = frequency
            graph_page.cosine_frame.amplitude = amplitude
            graph_page.cosine_frame.phase = phase
            graph_page.cosine_frame.update_plot()

            graph_page.pack(fill=tk.BOTH, expand=True)

        self.input_page.pack_forget()

    def switch_to_graph_page_index(self, index):
        if index < 0 or index >= len(self.graph_pages):
            return
        if self.current_page:
            self.current_page.pack_forget()
        self.current_page = self.graph_pages[index]
        self.current_page.pack(fill=tk.BOTH, expand=True)

    def switch_to_input_page(self):
        if self.current_page:
            self.current_page.pack_forget()
        self.input_page.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = SineWavePlotterApp()
    app.mainloop()
