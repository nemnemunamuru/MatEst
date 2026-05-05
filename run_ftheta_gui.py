if __name__ == '__main__':
    # defer importing the GUI class until runtime to reduce module import time
    from matest.gui import App
    App().mainloop()
