##################################################################################################################################
# CUSTOM CONDA ENVIRONMENT: 'pyWAXS' - need to generate .yml file for env reproduction.
# Jupyter Notebook Kernel: keithwhite@Keiths-MacBook-Pro/opt/anaconda3/envs/pyWAXS
# ----------------------------------------------------------------------------------------- #
# Contributors: Keith White, Zihan Zhang
# Toney Group, University of Colorado Boulder
# Updated: 04/21/2023
# Version Number: NSLS-II, Version 1.3
# Description: Bokeh plotting for 2D diffraction simulations.
##################################################################################################################################

# -- IMPORT LIBRARIES ------------------------------------------------ #
# import montecarlo_peaks as mcpeak
# import diffraction_script as diff
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import holoviews.plotting.bokeh
from holoviews import opts
from bokeh.models import ColumnDataSource, DataTable, TableColumn, StringFormatter, LabelSet, Div, CustomJS, Slider, Label, Image, ColorBar, LinearColorMapper, Scatter
from bokeh.layouts import row, column
from scipy.interpolate import RegularGridInterpolator
from bokeh.plotting import show, figure
from bokeh.io import output_notebook, push_notebook, show, curdoc, reset_output
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.util.compiler import TypeScript

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

hv.extension('bokeh')

slider_template = """
<div class="bk-slider-parent">
    <div class="bk-slider bk-bs-slider">
        <input type="range" class="bk-slider-input" id="{slider_id}" value="{slider_value}" min="{slider_start}" max="{slider_end}" step="{slider_step}">
        <div class="bk-slider-value">{slider_title}: {slider_value}</div>
    </div>
</div>
"""

def slider_to_html(slider):
    slider_id = slider.id
    slider_value = slider.value
    slider_start = slider.start
    slider_end = slider.end
    slider_step = slider.step
    slider_title = slider.title

    slider_html = slider_template.format(
        slider_id=slider_id,
        slider_value=slider_value,
        slider_start=slider_start,
        slider_end=slider_end,
        slider_step=slider_step,
        slider_title=slider_title,
    )
    return slider_html

def plot_heatmap(data, x_label='X-axis', y_label='Y-axis', z_label='Value', title='Heatmap',
                 font_size=12, tick_size=10, cmap='turbo', width=400, height=400,
                 x_extent=None, y_extent=None, plot_params=None):

    if isinstance(data, np.ndarray):
        if x_extent is None:
            x_extent = (0, data.shape[1])
        if y_extent is None:
            y_extent = (0, data.shape[0])

        X, Y = np.meshgrid(np.linspace(x_extent[0], x_extent[1], data.shape[1]),
                           np.linspace(y_extent[0], y_extent[1], data.shape[0]))
        df = pd.DataFrame({x_label: X.ravel(), y_label: Y.ravel(), z_label: data.ravel()})
    elif isinstance(data, pd.DataFrame):
        df = data.reset_index().melt(id_vars=[data.columns[0]], var_name=x_label, value_name=z_label)
        df = df.rename(columns={data.columns[0]: y_label})
    else:
        raise ValueError('Input data should be a 2D numpy array or a pandas DataFrame.')

    heatmap = df.hvplot.heatmap(x=x_label, 
                                y=y_label, 
                                C=z_label, 
                                colorbar=True, 
                                cmap=cmap, 
                                width=width, 
                                height=height, 
                                title=title)
                                # color_mapper=color_mapper)

    if plot_params is not None:
        xmin, xmax, ymin, ymax = plot_params['xmin'], plot_params['xmax'], plot_params['ymin'], plot_params['ymax']
        heatmap.opts(xlim=(xmin, xmax), ylim=(ymin, ymax))

    heatmap.opts(
        opts.HeatMap(
            xlabel=x_label, 
            ylabel=y_label, 
            colorbar_opts={'orientation': 'vertical', 'location': 'right'},
            fontsize={'title': font_size, 'labels': font_size, 'xticks': tick_size, 'yticks': tick_size}
        )
    )
    
    return heatmap

def plot_hklindex_bokeh(intensity_map, savepath, Mqxy, Mqz, FMiller, plot_params, imgParams, BPeakParams, table_width=400, table_height=400):
    # resolutionx, qxymax, qzmax, qzmin = imgParams
    # hkl_dimension = BPeakParams[2]
    theta_x, theta_y, hkl_dimension = BPeakParams["theta_x"], BPeakParams["theta_y"], BPeakParams["hkl_dimension"]
    # sigma_theta, sigma_phi, sigma_r = crystParams["sigma_theta"], crystParams["sigma_phi"], crystParams["sigma_r"]
    resolutionx, qxymax, qzmax, qzmin = imgParams["resolutionx"], imgParams["qxymax"], imgParams["qzmax"], imgParams["qzmin"]

    # ------------------------------------------------------------------------------------------------------------------------
    # Here is where the heatmap is created from the plot_heatmap() funciton above.
    x_extent = (-qxymax, qxymax)
    y_extent = (0, qzmax)

    scaleLog = plot_params.get("scaleLog")
    if scaleLog == True:
        intensity_map = np.log(intensity_map + 1)
    
    norm = plot_params.get("norm")
    if norm == True:
        gridMax = intensity_map.max()
        intensity_map = intensity_map/gridMax
    # x_label='$\mathregular{q_{xy}}$ ($\AA^{-1}$)', y_label='$\mathregular{q_z}$ ($\AA^{-1}$)'

    width = int(900)
    height = int(width * 0.67)

    x_label = "\[q_{xy}\,(Å^{-1})\]"
    y_label = "\[q_{z}\,(Å^{-1})\]"

    heatmap = plot_heatmap(intensity_map, 
                           x_label=x_label, 
                           y_label=y_label, 
                           z_label='Intensity',
                           title=plot_params['header'], 
                           font_size=plot_params['headerfontsize'], 
                           tick_size=plot_params['tickfontsize'],
                           cmap=plot_params['cmap'], 
                           width=width, 
                           height=height, 
                           x_extent=x_extent, 
                           y_extent=y_extent, 
                           plot_params=plot_params)

    # ------------------------------------------------------------------------------------------------------------------------
    # Here is where the (h k l) labels are generated, and their corresponding qxy/qz values are stored.
    Mindexrange = np.linspace(0, hkl_dimension, hkl_dimension+1)
    Mindexrange = Mindexrange.astype('int')
    simuposi = np.zeros([100,2])
    isimuposi = 0
    MaxI = 0
    label_data = []
    hkl_list = []
    qxy_list = []
    qz_list = []

    for h in Mindexrange:
        for k in Mindexrange:
            for l in Mindexrange:
                if Mqxy[h,k,l]<qxymax and Mqz[h,k,l]>qzmin and Mqz[h,k,l]<qzmax:
                    MaxI = np.maximum(FMiller[h,k,l], MaxI)

    for h in Mindexrange:
        for k in Mindexrange:
            for l in Mindexrange:
                if Mqxy[h,k,l]<qxymax and Mqz[h,k,l]>qzmin and Mqz[h,k,l]<qzmax:
                    if FMiller[h,k,l] > plot_params['hklcutoff']*MaxI:
                        simuposi[isimuposi,0] = Mqxy[h,k,l]
                        simuposi[isimuposi,1] = Mqz[h,k,l]
                        isimuposi = isimuposi+1
                        
                        textstr = '('+str(h-hkl_dimension)+','+str(l-hkl_dimension)+','+str(-k+hkl_dimension)+')'
                        hkl_list.append(textstr)
                        qxy_list.append(np.round(Mqxy[h,k,l],decimals = 2))
                        qz_list.append(np.round(Mqz[h,k,l], decimals = 2))
                        label_data.append({'x': Mqxy[h,k,l], 'y': Mqz[h,k,l], 'text': textstr})

    markers = hv.Scatter((simuposi[:, 0], simuposi[:, 1])).opts(size=6, color='red', tools=['hover']) # Create a Scatter plot for markers
    heatmap_with_markers = heatmap * markers # Combine the heatmap and markers

    label_df = pd.DataFrame(label_data) # Create a DataFrame for LabelSet

    label_source = ColumnDataSource(label_df) # Create LabelSet for annotations
    labels = LabelSet(x='x', 
                      y='y', 
                      text='text', 
                      source=label_source, 
                      x_offset=5, 
                      y_offset=5, 
                      render_mode='canvas', 
                      text_color = '#F6E9E6',
                      text_font_size = {'value': '12px'})

    # Create the Bokeh plot with the heatmap, markers, and labels - this merges the markers with the heatmap.
    bokeh_heatmap = hv.renderer('bokeh').get_plot(heatmap_with_markers).state
    bokeh_heatmap.add_layout(labels)

    # Remove gridlines from the heatmap
    bokeh_heatmap.xgrid.grid_line_color = None
    bokeh_heatmap.ygrid.grid_line_color = None
    
    # ------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------
    # Here is where the data table is created and populated.
    data_dict = {
    '(h k l)': hkl_list,
    'qxy': qxy_list,
    'qz': qz_list,
    'd(Å)': [''] * len(hkl_list)
    }

    table_columns = [
        TableColumn(field='(h k l)', 
                    title='(h k l)', 
                    formatter=StringFormatter(font_style='bold')),
        TableColumn(field='qxy', title='qxy'),
        TableColumn(field='qz', title='qz'),
        TableColumn(field='d(Å)', title='d(Å)'),
    ]
    
    data_table = DataTable(source=ColumnDataSource(data_dict), 
                       columns=table_columns,
                       autosize_mode = 'fit_columns',
                       width=table_width, 
                       height=table_height, 
                       css_classes=["custom_table"])
    # ------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------
    # Here is where the contrast slider widget objects are created and formatted.
    # Add the Div widget with a white background
    div_white_bg = Div(width=table_width, height=table_height - 50, style={'background-color': 'white'})  # Reduce the height

    # Create JavaScript callbacks for the contrast sliders
    js_callback_lower = CustomJS(args=dict(image_renderer=bokeh_heatmap.renderers[0], color_bar=bokeh_heatmap.select_one({'type': ColorBar})), code="""
        var new_value = cb_obj.value;
        var img = image_renderer.glyph;
        img.color_mapper.low = new_value;
        image_renderer.glyph.color_mapper = img.color_mapper;
        
        // Update the color bar
        color_bar.color_mapper.low = new_value;
        color_bar.color_mapper.trigger('change');
    """)

    js_callback_upper = CustomJS(args=dict(image_renderer=bokeh_heatmap.renderers[0], color_bar=bokeh_heatmap.select_one({'type': ColorBar})), code="""
        var new_value = cb_obj.value;
        var img = image_renderer.glyph;
        img.color_mapper.high = new_value;
        image_renderer.glyph.color_mapper = img.color_mapper;
        
        // Update the color bar
        color_bar.color_mapper.high = new_value;
        color_bar.color_mapper.trigger('change');
    """)

    contrast_low = Slider(start=0, end=intensity_map.max(), value=0, step=0.1, title="Low")
    contrast_high = Slider(start=0, end=intensity_map.max(), value=intensity_map.max(), step=0.1, title="High")
    color_mapper = LinearColorMapper(palette="Turbo256", low=contrast_low.value, high=contrast_high.value)

    # -- Callback function to update the heatmap based on contrast scaling sliders.
    def update(attr, old, new):
        color_mapper.low = contrast_low.value
        color_mapper.high = contrast_high.value
        push_notebook()

    # -- Connect the sliders to the update function.
    contrast_low.on_change('value', update)
    contrast_high.on_change('value', update)
    # ------------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------------
    # Interpolate intensity_map to create a smoother heatmap
    x = np.arange(intensity_map.shape[1])
    y = np.arange(intensity_map.shape[0])
    f = RegularGridInterpolator((y, x), intensity_map, method='linear', bounds_error=False, fill_value=None)

    x_new = np.linspace(0, len(x) - 1, len(x) * 10)  # Increase the factor from 5 to 10
    y_new = np.linspace(0, len(y) - 1, len(y) * 10)  # Increase the factor from 5 to 10

    yv, xv = np.meshgrid(y_new, x_new, indexing='ij')
    intensity_map_smooth = f((yv, xv))

    # Get the image glyph from the heatmap
    image_glyph = None
    for r in bokeh_heatmap.renderers:
        if isinstance(r.glyph, Image):
            image_glyph = r.glyph
            break

    # Update the image glyph data source with the intensity_map_smooth
    if image_glyph:
        image_glyph.data_source.data['image'] = [intensity_map_smooth]
    # ------------------------------------------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------------------------------------------
    # Add custom CSS to set the header background color to white
    css = Div(text="""
    <style>
        .bk-root .slick-header-column {
            background-color: white !important;
        }
    </style>
    """)

    # ------------------------------------------------------------------------------------------------------------------------
    # Here is where the layout generation and formatting occurs.
    # layout = column(row(bokeh_heatmap, contrast_low, contrast_high, div_white_bg), data_table)
    layout = row(column(bokeh_heatmap, contrast_low, contrast_high, div_white_bg), data_table)
    layout.children.insert(0, css) # implement the css white header background 

    # -- Set up the custom CSS.
    custom_css = """
    <style>
        table.index_table {
            border-collapse: collapse;
            font-family: "Lucida Console", Monaco, monospace;
            font-size: 12px;
        }
        table.index_table td {
            border: 1px solid black;
            padding: 2px 4px;
            text-align: center;
        }
    </style>
    """

    # Add the custom CSS style to your HTML head
    bokeh_plot_html = file_html(layout, CDN, "Bokeh Plot")
    bokeh_plot_html = bokeh_plot_html.replace("</head>", f"{custom_css}</head>")

    reset_output()

    show(layout) # display the output, ported out to HTML in your browser with the prescribed formatting conditions.

    return layout

'''
#     # -- Heatmap plot generation.
# def plot_heatmap(intensity_map, x_label, y_label, x_extent, y_extent, plot_params):
#         p = figure(x_range=x_extent, 
#                    y_range=y_extent, 
#                    tools="box_zoom,wheel_zoom,reset,save", 
#                    **plot_params)
        
#         p.image(image=[intensity_map], 
#                 x=x_extent[0], 
#                 y=y_extent[0], 
#                 dw=x_extent[1] - x_extent[0], 
#                 dh=y_extent[1] - y_extent[0], 
#                 color_mapper=color_mapper)
        
#         p.xaxis.axis_label = x_label
#         p.yaxis.axis_label = y_label
#         p.toolbar.logo = None
#         return p
'''

'''
    # Create the sliders
    # contrast_low = Slider(title="Lower Intensity Limit", start=0, end=intensity_map.data.max(), value=low.value, step=1, width=slider_width)
    # contrast_high = Slider(title="Upper Intensity Limit", start=0, end=intensity_map.data.max(), value=high.value, step=1, width=slider_width)

    # Define the update function
    # def update(attr, old, new):
    #     color_mapper.low = contrast_low.value
    #     color_mapper.high = contrast_high.value
    #     intensity_map.data = dict(image=[intensity_map.data], x=[Mqxy.min()], y=[Mqz.min()], dw=[Mqxy.max() - Mqxy.min()], dh=[Mqz.max() - Mqz.min()])

    # Add the callbacks for the sliders
    # contrast_low.on_change('value', update)
    # contrast_high.on_change('value', update)

    # # Add Slider widgets for contrast scaling
    # contrast_slider_lower = Slider(start=intensity_map.min(), end=intensity_map.max(), value=intensity_map.min(), step=1, title="Lower Contrast Scaling")
    # contrast_slider_upper = Slider(start=intensity_map.min(), end=intensity_map.max(), value=intensity_map.max(), step=1, title="Upper Contrast Scaling")
    # contrast_slider_lower.js_on_change('value', js_callback_lower)
    # contrast_slider_upper.js_on_change('value', js_callback_upper)

    # Create a custom CSS style 
    # custom_css = """
    # <style>
    #     .rotated-slider {
    #         transform: rotate(90deg);
    #         transform-origin: left top;
    #         margin-top: 40px;
    #     }
    # </style>
    # """

    # contrast_low.css_classes = ["rotated-slider"]
    # contrast_high.css_classes = ["rotated-slider"]

    # Wrap the sliders in Div elements and apply a 90-degree rotation
    # slider_wrapper_lower = Div(text=f"""
    # <div style="transform: rotate(90deg); transform-origin: left top; margin-top: 40px;">
    #     {slider_to_html(contrast_slider_lower)}
    # </div>
    # """)

    # slider_wrapper_upper = Div(text=f"""
    # <div style="transform: rotate(90deg); transform-origin: left top; margin-top: 40px;">
    #     {slider_to_html(contrast_slider_upper)}
    # </div>
    # """)

    # -- Set up the contrast scaling bars.
    # contrast_low = Slider(start=0, end=np.log(intensity_map.max() + 1), value=0, step=0.1, title="Low")
    # contrast_high = Slider(start=0, end=np.log(intensity_map.max() + 1), value=np.log(intensity_map.max() + 1), step=0.1, title="High")
    '''

'''
# TS_CODE = """
# import * as p from "core/properties"
# import {Label, LabelView} from "models/annotations/label"
# declare const katex: any

# export class LatexLabelView extends LabelView {
#   model: LatexLabel

#   render(): void {
#     //--- Start of copied section from ``Label.render`` implementation

#     // Here because AngleSpec does units tranform and label doesn't support specs
#     let angle: number
#     switch (this.model.angle_units) {
#       case "rad": {
#         angle = -this.model.angle
#         break
#       }
#       case "deg": {
#         angle = (-this.model.angle * Math.PI) / 180.0
#         break
#       }
#       default:
#         throw new Error("unreachable code")
#     }

#     const panel = this.layout ?? this.plot_view.layout.center_panel

#     let sx = this.model.x_units == "data" ? this.coordinates.x_scale.compute(this.model.x) : panel.xview.compute(this.model.x)
#     let sy = this.model.y_units == "data" ? this.coordinates.y_scale.compute(this.model.y) : panel.yview.compute(this.model.y)

#     sx += this.model.x_offset
#     sy -= this.model.y_offset

#     //--- End of copied section from ``Label.render`` implementation
#     // Must render as superpositioned div (not on canvas) so that KaTex
#     // css can properly style the text
#     this._css_text(this.layer.ctx, "", sx, sy, angle)

#     // ``katex`` is loaded into the global window at runtime
#     // katex.renderToString returns a html ``span`` element
#     katex.render(this.model.text, this.el, {displayMode: true})
#   }
# }

# export namespace LatexLabel {
#   export type Attrs = p.AttrsOf<Props>

#   export type Props = Label.Props
# }

# export interface LatexLabel extends LatexLabel.Attrs {}

# export class LatexLabel extends Label {
#   properties: LatexLabel.Props
#   __view_type__: LatexLabelView

#   constructor(attrs?: Partial<LatexLabel.Attrs>) {
#     super(attrs)
#   }

#   static {
#     this.prototype.default_view = LatexLabelView
#   }
# }
# """

# class LatexLabel(Label):
#     """A subclass of the Bokeh built-in `Label` that supports rendering
#     LaTex using the KaTex typesetting library.

#     Only the render method of LabelView is overloaded to perform the
#     text -> latex (via katex) conversion. Note: ``render_mode="canvas``
#     isn't supported and certain DOM manipulation happens in the Label
#     superclass implementation that requires explicitly setting
#     `render_mode='css'`).
#     """
#     __javascript__ = ["https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.js"]
#     __css__ = ["https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.css"]
#     __implementation__ = TypeScript(TS_CODE)
'''