import tkinter as tk
from tkinter import filedialog, messagebox
import os

class BurmeseAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Burmese Text Annotator")
        self.root.geometry("800x600")

        # Predefined tag categories with key bindings
        self.tags = {
            "1": "person",
            "2": "organization",
            "3": "place",
            "4": "event",
            "5": "datetime",
            "6": "nounword",
            "7": "verbword",
            "8": "particle",
            "9": "punctuation",
            "0": "OOA"
        }

        # Create UI elements
        self.create_ui()

        # Bind keys for tagging
        for key in self.tags:
            self.root.bind(f"<Key-{key}>", self.annotate_selection)

    def create_ui(self):
        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5, fill=tk.X)

        # Load and Save buttons
        tk.Button(button_frame, text="Load File", command=self.load_file).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save File", command=self.save_file).pack(side=tk.LEFT, padx=5)

        # Text canvas for editing
        self.text_canvas = tk.Text(self.root, wrap=tk.WORD, height=30, width=80, font=("Zawgyi-One", 12))
        self.text_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Label for instructions
        instructions = "Select text and press: " + ", ".join(f"{k}: {v}" for k, v in self.tags.items())
        tk.Label(self.root, text=instructions, font=("Arial", 10)).pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_canvas.delete(1.0, tk.END)
                self.text_canvas.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    content = self.text_canvas.get(1.0, tk.END).strip()
                    f.write(content)
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    def annotate_selection(self, event):
        # Get the pressed key (e.g., '1', '2')
        tag = self.tags.get(event.keysym)
        if not tag:
            return

        try:
            # Get selected text
            selected_text = self.text_canvas.selection_get()
            if selected_text:
                # Get selection range
                start = self.text_canvas.index(tk.SEL_FIRST)
                end = self.text_canvas.index(tk.SEL_LAST)

                # Replace selected text with tagged version
                tagged_text = f"<{tag}>{selected_text}</{tag}>"
                self.text_canvas.delete(start, end)
                self.text_canvas.insert(start, tagged_text)

                # Clear selection
                self.text_canvas.tag_remove(tk.SEL, 1.0, tk.END)
        except tk.TclError:
            # No selection made
            pass

def main():
    root = tk.Tk()
    app = BurmeseAnnotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()