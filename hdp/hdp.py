#!/usr/bin/env python
"""
hdp.py

Heatwave Diagnostics Package (HDP)

Entry point for package.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
"""
from hdp.graphics.notebook import HDPNotebook
from hdp.graphics.figure import *
from hdp.utils import get_func_description
from tqdm.auto import tqdm


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