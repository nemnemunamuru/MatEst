import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
# Heavy plotting / data libraries are imported lazily in _ensure_plot_ready
from .io import read_table_with_type
from .estimator import estimate


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MatEst — F-theta Estimator')
        # Set window to ~80% of 1980x1280 → 1584x1024 and disable resizing
        self.geometry('1584x1024')
        self.resizable(False, False)

        # Configure a blue-themed ttk style for frames, labels, and buttons
        try:
            style = ttk.Style(self)
            style.theme_use('clam')
            style.configure('Blue.TFrame', background='#eaf4ff')
            style.configure('Blue.TLabel', background='#eaf4ff', foreground='#072b5b', font=('TkDefaultFont', 11))
            style.configure('Blue.TButton', background='#1976d2', foreground='white')
            style.map('Blue.TButton', background=[('active', '#1565c0')])
            self.configure(background='#eaf4ff')
        except Exception:
            pass
        self.csv_path = tk.StringVar()
        self.method = tk.StringVar(value='AxisRadial+RBF')
        self.degree = tk.IntVar(value=3)
        self.lam = tk.DoubleVar(value=1e-5)
        self.sigma = tk.DoubleVar(value=10.0)
        self.dataset_type = tk.StringVar(value='')
        self.df_in = None
        self.df_out = None
        self.stats = None
        self._build_widgets()

    def _build_widgets(self):
        frm = ttk.Frame(self, style='Blue.TFrame')
        frm.pack(side='top', fill='x', padx=12, pady=12)
        r = 0
        ttk.Label(frm, text='Input CSV:', style='Blue.TLabel').grid(row=r, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.csv_path, width=82).grid(row=r, column=1, sticky='we', padx=6)
        ttk.Button(frm, text='Browse…', command=self.browse, style='Blue.TButton').grid(row=r, column=2)
        r += 1

        ttk.Label(frm, text='Method:', style='Blue.TLabel').grid(row=r, column=0, sticky='w')
        cb = ttk.Combobox(frm, textvariable=self.method, state='readonly', width=18,
                          values=['Polynomial','Linear','RBF','Radial','AxisRadial','AxisRadial+RBF'])
        cb.grid(row=r, column=1, sticky='w', padx=6)
        cb.bind('<<ComboboxSelected>>', lambda e: self._toggle_params())
        r += 1

        self.lbl_deg = ttk.Label(frm, text='Max degree (poly/radial only):', style='Blue.TLabel')
        self.lbl_deg.grid(row=r, column=0, sticky='w')
        self.spn_deg = ttk.Spinbox(frm, from_=1, to=9, textvariable=self.degree, width=6)
        self.spn_deg.grid(row=r, column=1, sticky='w', padx=6)
        r += 1

        ttk.Label(frm, text='lambda (ridge):', style='Blue.TLabel').grid(row=r, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.lam, width=12).grid(row=r, column=1, sticky='w', padx=6)
        r += 1

        self.lbl_sigma = ttk.Label(frm, text='sigma (RBF):', style='Blue.TLabel')
        self.lbl_sigma.grid(row=r, column=0, sticky='w')
        self.ent_sigma = ttk.Entry(frm, textvariable=self.sigma, width=12)
        self.ent_sigma.grid(row=r, column=1, sticky='w', padx=6)
        r += 1

        info_row = ttk.Frame(frm, style='Blue.TFrame')
        info_row.grid(row=r, column=0, columnspan=3, sticky='we')
        self.lbl_info = ttk.Label(info_row, text='No file loaded', style='Blue.TLabel')
        self.lbl_info.pack(side='left')
        self.lbl_fmt = ttk.Label(info_row, text='', font=('TkDefaultFont', 10, 'bold'), style='Blue.TLabel')
        self.lbl_fmt.pack(side='left', padx=12)
        r += 1

        btns = ttk.Frame(frm, style='Blue.TFrame')
        btns.grid(row=r, column=0, columnspan=3, sticky='w', pady=(8,0))
        ttk.Button(btns, text='Load', command=self.load_csv, style='Blue.TButton').pack(side='left', padx=4)
        ttk.Button(btns, text='Estimate + Plot', command=self.run_estimate, style='Blue.TButton').pack(side='left', padx=4)
        ttk.Button(btns, text='Export with labels…', command=self.export_with_labels_csv, style='Blue.TButton').pack(side='left', padx=4)

        for i in range(3):
            frm.columnconfigure(i, weight=1)
        self._toggle_params()

        # --- Plot area (bottom, smaller) ---
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side='bottom', fill='both', expand=True, padx=12, pady=8)
        self.plot_frame = plot_frame

        # plotting area will be initialized lazily to speed startup
        self._plot_initialized = False
        # placeholder label until plotting components are created
        self._plot_placeholder = ttk.Label(self.plot_frame, text='Plots will appear here after Estimate', style='Blue.TLabel')
        self._plot_placeholder.pack(fill='both', expand=True)

    def _toggle_params(self):
        m = self.method.get()
        if m == 'Linear':
            self.spn_deg.configure(state='disabled')
            self.lbl_deg.configure(foreground='gray')
            self.ent_sigma.configure(state='disabled')
            self.lbl_sigma.configure(foreground='gray')
        elif m in ('Polynomial','Radial','AxisRadial','AxisRadial+RBF'):
            self.spn_deg.configure(state='normal')
            self.lbl_deg.configure(foreground='black')
            self.ent_sigma.configure(state='disabled')
            self.lbl_sigma.configure(foreground='gray')
        else:  # RBF
            self.spn_deg.configure(state='disabled')
            self.lbl_deg.configure(foreground='gray')
            self.ent_sigma.configure(state='normal')
            self.lbl_sigma.configure(foreground='black')

    def _ensure_plot_ready(self):
        """Create plotting objects and import heavy libraries on demand."""
        if self._plot_initialized:
            return
        # remove placeholder
        try:
            self._plot_placeholder.destroy()
        except Exception:
            pass
        # import heavy libraries now
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from .plotting import make_four_plots

        # create figure and canvas
        self.fig = Figure(figsize=(12, 4.5), dpi=120)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        try:
            self.toolbar.pack(side='bottom', fill='x')
            self.toolbar.update()
        except Exception:
            pass

        # draw initial empty figure
        try:
            make_four_plots(pd.DataFrame(columns=['Ideal_X','Ideal_Y','Real_X','Real_Y','Pred_Real_X','Pred_Real_Y']),
                            {'n_total':0,'n_measured':0,'method':'','lambda':0,'degree_used':None,'degree_requested':None,'rmse_x':None,'rmse_y':None,'sigma':None},
                            dataset_type='', fig=self.fig)
            self.canvas.draw()
        except Exception:
            # if plotting fails, leave placeholder removed but don't crash
            pass

        self._plot_initialized = True

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files','*.csv'), ('All files','*.*')])
        if path:
            self.csv_path.set(path)

    def load_csv(self):
        path = self.csv_path.get()
        if not path:
            messagebox.showwarning('Warning', 'Select a CSV file.')
            return
        try:
            df, fmt = read_table_with_type(path)
        except Exception as e:
            messagebox.showerror('Read error', f'Failed to read CSV:\n{e}')
            return
        self.df_in = df
        self.dataset_type.set(fmt)
        n_meas = (df['Real_X'].notna() & df['Real_Y'].notna()).sum()
        self.lbl_info.config(text=f'Loaded: {os.path.basename(path)}  N={len(df)} / Measured={n_meas}')
        if fmt == 'OCT':
            self.lbl_fmt.config(text='OCT補正マップ', foreground='green')
            self.method.set('Linear')
        else:
            self.lbl_fmt.config(text='歪補正', foreground='red')
            self.method.set('AxisRadial+RBF')
            self.degree.set(3)
        self._toggle_params()
        # prepare plotting (lazy) so the UI becomes responsive before heavy imports
        self._ensure_plot_ready()

    def run_estimate(self):
        if self.df_in is None:
            messagebox.showwarning('Warning', 'Load CSV first.')
            return
        m = self.method.get()
        d = int(self.degree.get())
        lam = float(self.lam.get())
        sigma = float(self.sigma.get())
        try:
            out, stats = estimate(self.df_in, m, d, lam, sigma)
        except Exception as e:
            messagebox.showerror('Estimate error', f'Error in estimation:\n{e}')
            return
        self.df_out = out
        self.stats = stats

        # update embedded figure
        self._ensure_plot_ready()
        # now safe to import and call plotting
        from .plotting import make_four_plots
        make_four_plots(out, stats, dataset_type=self.dataset_type.get(), fig=self.fig)
        self.canvas.draw()

        msg = f"Done: method={stats['method']} / Measured={stats['n_measured']}"
        if stats['degree_used'] is not None:
            msg += f" / degree_used={stats['degree_used']}"
        if stats['rmse_x'] is not None:
            msg += f"  RMSE(X)={stats['rmse_x']:.4g}, RMSE(Y)={stats['rmse_y']:.4g}"
        if stats.get('axis_class'):
            xlab, ylab = stats['axis_class']
            msg += f"  Axis: X={xlab}, Y={ylab}"
        if stats.get('rotation_deg') is not None:
            msg += f"  Rot={stats['rotation_deg']:.3f}°"
        if self.dataset_type.get() == 'OCT':
            msg += "  View=OCT Δ-mode"
        self.lbl_info.config(text=msg)

    def export_with_labels_csv(self):
        if self.df_out is None:
            messagebox.showwarning('Warning', 'Run estimation first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files','*.csv'), ('All files','*.*')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if self.dataset_type.get() == 'OCT':
                    writer.writerow(['MeasurePosX','MeasurePosY','CorrPosX','CorrPosY'])
                    for X, Y, PX, PY in zip(self.df_out['Ideal_X'], self.df_out['Ideal_Y'],
                                             self.df_out['Pred_Real_X'], self.df_out['Pred_Real_Y']):
                        writer.writerow([X, Y, PX - X, PY - Y])
                else:
                    f.write('//Ideal_X,Ideal_Y,Real_X,Real_Y\n')
                    for X, Y, PX, PY in zip(self.df_out['Ideal_X'], self.df_out['Ideal_Y'],
                                             self.df_out['Pred_Real_X'], self.df_out['Pred_Real_Y']):
                        writer.writerow([X, Y, PX, PY])
        except Exception as e:
            messagebox.showerror('Save error', f'Failed to save:\n{e}')
            return
        messagebox.showinfo('Saved', f'Wrote:\n{path}')
