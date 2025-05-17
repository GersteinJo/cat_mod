from cat_mod import SEP
from residual.loader import ImageLabelDataset

import numpy as np
import torch
from scipy.stats import pearsonr

# for plotting 
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.palettes import Viridis256
from bokeh.layouts import row

import plotly.express as px

def train_classifier(config, peck_obs_num = 200, n_iter = 10000, encoded_data=None, labels=None, logger = None):
    shuffle_mask = np.arange(len(labels))
    np.random.shuffle(shuffle_mask)

    sep = SEP.SEP(config['num_exemp'], output_space_shape=2, input_space_shape = 64,
    lr = config['lr'], dr=config['dr'], omega = config['omega'], delta = config['delta'])
    results = np.zeros(peck_obs_num)
    peck = [1]
    counter = 0

    assert n_iter < len(labels), "Number of iterations should not surpass number of datapoints!"
    for i in range(n_iter):
        
        obs, label = encoded_data[shuffle_mask[i]].reshape(1,-1), labels[shuffle_mask[i]:shuffle_mask[i]+1]
        # print(label)
        peck = sep.predict(obs)

        if peck[0]==1:# and not (logger is None):
            # print("HEHEHEHEHEHHHHEHHEHEHEHEHEHEH")
            if logger:
                logger.log(
                    {"peck_edible": label[0]},
                    step = counter
                    )
            results[counter] = label.item()
            counter += 1
            if counter == peck_obs_num: print(f"Trial {i} finished!"); break
            sep.fit(obs, label)
        elif np.random.random() < 0.05:
            # print(f"label {label}")
            sep.fit(obs, label)


def encode_dataset(encoder, dataloader, orig_loader, device='cpu'):
    """
    Encodes images and labels using the provided encoder (e.g., a CNN or transformer).
    
    Args:
        encoder: encoder function(!) not an encoder itself
        dataloader: PyTorch DataLoader yielding (image_tensor, label) pairs.
        device: 'cpu' or 'cuda'
        
    Returns:
        Tuple of numpy arrays (encoded_features_numpy, labels_numpy, original_images_numpy)
    """
    all_embeddings = []
    all_labels = []
    all_original_images = []

    with torch.no_grad():
        for (images_dino, targets), (images_orig, _) in zip(loader, original_loader):
            images_dino = images_dino.to(device, non_blocking=True)
            embeddings = encoder_function(images_dino).cpu()
            all_embeddings.append(embeddings)
            all_labels.append(targets)
            all_original_images.append(images_orig)

    embeddings = torch.cat(all_embeddings)  # [N, D]
    labels = torch.cat(all_labels)
    original_images = torch.cat(all_original_images)
    original_images = original_images.reshape(original_images.shape[0], -1)

    return (embeddings.numpy(),
            labels.numpy(),
            original_images.numpy())

    

def compare_embeddings(emb1, emb2):
    dissim1 = 1. - np.corrcoef(emb1)
    dissim2 = 1. - np.corrcoef(emb2)

    triu_indices = np.triu_indices_from(dissim1, k=1)
    flat1 = dissim1[triu_indices]
    flat2 = dissim2[triu_indices]

    # Compute second-order similarity (Pearson correlation)
    r, _ = pearsonr(flat1, flat2)
    return r

def embedding_plotter(embedding, data=None, hue=None, hover=None, tools = None, nv_cat = 5, height = 400, width = 400, display_result=True):
    '''
    Рисовалка эмбеддинга. 2D renderer: bokeh. 3D renderer: plotly.
    Обязательные инструменты:
        - pan (двигать график)
        - box zoom
        - reset (вылезти из зума в начальное положение)

        embedding: something 2D/3D, slicable ~ embedding[:, 0] - валидно
            Эмбеддинг
        data: pd.DataFrame
            Данные, по которым был построен эмбеддинг
        hue: string
            Колонка из data, по которой красим точки. Поддерживает интерактивную легенду: по клику на каждое
                значение hue можно скрыть весь цвет.
        hover: string or list of strings
            Колонк[а/и] из data, значения которых нужно выводить при наведении мышки на точку
        nv_cat: int
            number of unique values to consider column categorical
        tools: iterable or string in form "tool1,tool2,..." or ["tool1", "tool2", ...]
            tools for the interactive plot
        height, width: int
            parameters of the figure
        display_result: boolean
            if the results are displayed or just returned

    '''
    if tools is None:
        tools = 'lasso_select,box_select,pan,zoom_in,zoom_out,reset,hover'
    else:
        if hover and not("hover" in tools):
            tools = 'hover,'+",".join(tools)


    if embedding.shape[1] == 3:
        if hover:
            hover_data = {h:True for h in hover}
        else:
            hover_data = None
        df = pd.DataFrame(embedding, columns = ['x', 'y', 'z'])
        df = pd.concat((df, data), axis=1)
        fig = px.scatter_3d(
            data_frame = df,
            x='x',
            y='y',
            z='z',
            color=df[hue],
            hover_data = hover_data
        )

        fig.update_layout(
            modebar_add=tools.split(","),
        )

        fig.update_traces(marker_size=1, selector=dict(type='scatter3d'))

        if display_result: fig.show()

    if embedding.shape[1] == 2:
        output_notebook()
        df = pd.DataFrame(embedding, columns = ['x', 'y'])
        df = pd.concat((df, data), axis=1)
        tooltips = [
            ('x, y', '$x, $y'),
            ('index', '$index')
        ]
        if hover:
            for col in hover:
                tooltips.append((col, "@"+col))
        fig = figure(tools=tools, width=width, height=height, tooltips=tooltips)
        if df[hue].nunique() < nv_cat or df[hue].dtype == "category":
            df[hue] = df[hue].astype(str)
            source = ColumnDataSource(df)
            color_mapper = factor_cmap(
            field_name=hue,
            palette='Category10_3',
            factors=df[hue].unique()
            )
            fig.scatter(
            x='x', y='y',
            color=color_mapper,
            source=source,
            legend_group=hue)

            fig.legend.location = 'bottom_left'
            fig.legend.click_policy = 'mute'
        else:
            source = ColumnDataSource(df)
            color_mapper = linear_cmap(
                field_name=hue,
                palette=Viridis256,
                low=min(df[hue]),
                high=max(df[hue]))
            fig.scatter(
                x='x', y='y',
                color=color_mapper,
                source=source)
            color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0,0), title = hue)
            fig.add_layout(color_bar, 'right')


        if display_result: show(fig)

    if embedding.shape[1] > 3:
        print("wrong species, doooooodes")
    else: return fig
