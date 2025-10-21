import matplotlib.pyplot as plt
import base64
import nbformat
from io import BytesIO
from datetime import datetime
from hdp.utils import get_func_description
from hdp.graphics.figure import *
from tqdm.auto import tqdm


class HDPNotebook():
    def create_section(self, section_name, label=None, rank=1, label_hidden=False):
        if label_hidden:
            label = None
        elif label is None:
            label = section_name

        self.__sections[section_name] = {
            "cells": [],
            "rank": rank,
            "label": label
        }
    
    
    def __init__(self):
        self.__sections = {}
        self.__num_figs = 0

    
    def add_markdown_cell(self, cell_data, section_name):
        if section_name not in self.__sections:
            self.create_section(section_name)
        self.__sections[section_name]["cells"].append(nbformat.v4.new_markdown_cell(cell_data))
    
    
    def add_figure_cell(self, figure, section_name=None, alt_text="Figure"):
        img_buffer = BytesIO()
        figure.savefig(img_buffer, format='png')
        img_buffer.seek(0)
    
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        cell_data = f'![{alt_text}](data:image/png;base64,{img_base64})'
        img_buffer.close()
        
        self.add_markdown_cell(cell_data=cell_data, section_name=section_name)
        self.__num_figs += 1

    
    def set_section_label(section_name, section_label):
        self.__sections[section_name]["label"] = section_label
    

    def _format_section_label(self, label):
        return f"## {label}"
    
    
    def save_notebook(self, path, title=None):
        from hdp.utils import get_version, get_time_stamp
        header_data = [
            '# Heatwave Diagnostics Package (HDP) Standard Figure Deck\n',
            '' if title is None else f'## Deck Title: {title}\n',
            '\n',
            '```\n',
            'Webpage: https://github.com/AgentOxygen/HDP\n',
            f'Version: HDP {get_version()}\n',
            f'Generation Timestamp: {get_time_stamp()}\n',
            f'Figures Generated: {self.__num_figs}\n',
            '```\n'
        ]
        
        self.create_section("header", rank=0, label_hidden=True)
        self.add_markdown_cell(header_data, "header")
        
        notebook_node = nbformat.v4.new_notebook()
        ranked_cells = {}
        for section_name in self.__sections:
            rank = self.__sections[section_name]["rank"]
            if rank in ranked_cells:
                ranked_cells[rank].append(section_name)
            else:
                ranked_cells[rank] = [section_name]

        ranks = list(ranked_cells.keys())
        ranks.sort()

        for rank in ranks:
            for section_name in ranked_cells[rank]:
                label = self.__sections[section_name]["label"]
                if label is not None:
                    label_cell = nbformat.v4.new_markdown_cell(self._format_section_label(label), metadata={"jp-MarkdownHeadingCollapsed": True})
                    
                    notebook_node.cells.append(label_cell)
                
                for cell in self.__sections[section_name]["cells"]:   
                    notebook_node.cells.append(cell)
        
        with open(path, 'w') as nb_file:
            nbformat.write(notebook_node, nb_file)


def create_notebook(hw_ds):
    assert "hdp_type" in hw_ds.attrs, "Missing 'hdp_type' attribute."

    notebook = HDPNotebook()
    
    if hw_ds.attrs["hdp_type"] == "measure":
        pass
    elif hw_ds.attrs["hdp_type"] == "threshold":
        pass
    elif hw_ds.attrs["hdp_type"] == "metric":
        index = 1
        
        section_name = f"Figures {index}"
        notebook.create_section(section_name)
        desc = get_func_description(plot_multi_measure_metric_comparisons)
        notebook.add_markdown_cell(f"### Figure {index}.2 \n{desc}", section_name)
        notebook.add_figure_cell(plot_multi_measure_metric_comparisons(hw_ds), section_name, alt_text=f"{section_name}")
        
        index += 1
        for metric in tqdm(list(hw_ds.data_vars), desc="Generating figures:"):
            section_name = f"Figures {index}-{metric}"
            
            notebook.create_section(section_name)
            notebook.add_markdown_cell("Description of these figures.", section_name)
            
            desc = get_func_description(plot_metric_parameter_comparison)
            notebook.add_markdown_cell(f"### Figure {index}.1 \n{desc}", section_name)
            notebook.add_figure_cell(plot_metric_parameter_comparison(hw_ds[metric]), section_name, alt_text=f"{section_name}")
            
            desc = get_func_description(plot_metric_timeseries)
            notebook.add_markdown_cell(f"### Figure {index}.2 \n{desc}", section_name)
            notebook.add_figure_cell(plot_metric_timeseries(hw_ds[metric]), section_name, alt_text=f"{section_name}")

            iindex = 3
            for fig in plot_metric_decadal_maps(hw_ds[metric]):
                desc = get_func_description(plot_metric_decadal_maps)
                notebook.add_markdown_cell(f"### Figure {index}.{iindex} \n{desc}", section_name)
                notebook.add_figure_cell(fig, section_name, alt_text=f"{section_name}")
                iindex += 1
            index += 1
    else:
        raise ValueError(f"Unexpected value for 'hdp_type' attribute, '{hw_ds.attrs["hdp_type"]}' is not 'measure', 'threshold', or 'metric'.")

    return notebook