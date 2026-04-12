import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

def to_long_3(df):
    df_long = df.melt(id_vars='backward', var_name='feature', value_name='value')
    df_long.rename(columns={'backward': 'step'}, inplace=True)
    return df_long


@dataclass
class PlotSettings:
    """
    Настройки для графика plot_styled_linechart
    """
    figsize: Tuple[int, int] = (8, 5)
    dpi: int = 120
    palette: Dict[str, str] = None
    linewidth_base: int = 2
    markersize_base: int = 6
    grid_alpha: float = 0.5
    legend_ncol: int = 4
    font_family: str = 'Arial'
    font_size: int = 10
    xlim_min: Optional[float] = None  # Минимальная граница по оси X
    xlim_max: Optional[float] = None  # Максимальная граница по оси X
    ylim_min: Optional[float] = None  # Минимальная граница по оси Y
    ylim_max: Optional[float] = None  # Максимальная граница по оси Y
    legend: bool = True
    y_step: int = 10
    x_step: int = 2000
    xlabel: str = 'backward'
    ylabel: str = 'Gap (%)'

    # def __post_init__(self):
    #     """Инициализация значений по умолчанию после создания объекта."""
    #     if self.palette is None:
    #         self.palette = {
    #             'rank_1': '#1f77b4',
    #             'rank_2': '#ff7f0e',
    #             'rank_3': '#2ca02c',
    #             'rank_4': '#2ca02c'
    #         }

def plot_styled_linechart(
    df: pd.DataFrame,
    title: str = "Динамика показателей за период",
    settings: PlotSettings = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Строит стилизованный линейный график с использованием Seaborn и hue.

    Параметры:
    df (pd.DataFrame): DataFrame с колонками ['feature', 'step', 'value']
    title (str): заголовок графика
    settings (PlotSettings): объект с настройками визуализации
    """

    # Если настройки не переданы — создаём стандартные
    if settings is None:
        settings = PlotSettings()

    # Настройка глобальных параметров шрифтов
    plt.rcParams['font.family'] = settings.font_family
    plt.rcParams['font.size'] = settings.font_size

    # Инициализация фигуры
    fig, ax = plt.subplots(figsize=settings.figsize, dpi=settings.dpi)

    # Установка палитры Seaborn
    #sns.set_palette(list(settings.palette.values()))

    # Построение графика с hue
    sns.lineplot(
        data=df,
        x='step',
        y='value',
        hue='feature',
        palette=settings.palette,
        linewidth=settings.linewidth_base,
        marker='o',
        markersize=settings.markersize_base,
        ax=ax,
        legend=settings.legend
    )

    # Ручная настройка стилей для конкретных категорий (если нужно)
    lines = ax.get_lines()
    for i, feature in enumerate(df['feature'].unique()):
        if feature == 'Rank 3' or feature == 'Rank 4':
            lines[i].set_linewidth(1)
            lines[i].set_markersize(4)
            lines[i].set_alpha(0.75)
            if feature == 'Rank 3' or feature == 'Rank 4':
                lines[i].set_linestyle('--')

    # Настройка осей
    ax.set_xlabel('Step', fontsize=12, color='#333333')
    ax.set_ylabel('Value', fontsize=12, color='#333333')

    # Установка границ осей, если они переданы
    if settings.xlim_min is not None or settings.xlim_max is not None:
        current_xlim = list(ax.get_xlim())
        if settings.xlim_min is not None:
            current_xlim[0] = settings.xlim_min
        if settings.xlim_max is not None:
            current_xlim[1] = settings.xlim_max
        ax.set_xlim(current_xlim)

    if settings.ylim_min is not None or settings.ylim_max is not None:
        current_ylim = list(ax.get_ylim())
        if settings.ylim_min is not None:
            current_ylim[0] = settings.ylim_min
        if settings.ylim_max is not None:
            current_ylim[1] = settings.ylim_max
        ax.set_ylim(current_ylim)
    else:
        y_min, y_max = ax.get_ylim()
        y_range = max(60, y_max)
        ax.set_ylim(-3, y_range)

    # Шаг делений оси
    ax.yaxis.set_major_locator(MultipleLocator(settings.y_step))
    ax.xaxis.set_major_locator(MultipleLocator(settings.x_step))

    # Сетка
    ax.grid(True, axis='y', linestyle='-', alpha=settings.grid_alpha)
    ax.grid(True, axis='x', linestyle='-', alpha=settings.grid_alpha) 

    # Горизонтальная линия у нуля
    ax.axhline(y=0, color='black', linewidth=1)

    if settings.legend:
        # Легенда
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=settings.legend_ncol,
            fontsize=10,
            frameon=True,
            fancybox=False,
            edgecolor='black',
            framealpha=0.2
        )
        legend.get_frame().set_linewidth(0.5)

    # Заголовок
    ax.set_title(
        title,
        fontsize=14,
        fontweight='bold',
        color='#000000',
        pad=15
    )

    # Лейблы
    ax.set_xlabel(settings.xlabel, fontsize=14)
    ax.set_ylabel(settings.ylabel, fontsize=14)

    # Фон
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Границы осей
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(3)
        ax.spines[spine].set_color('#222222')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Параметры тиков на осях
    ax.tick_params(
        axis='both',
        which='major',
        width=2,
        length=5,
        color='black',
        direction='out',
        pad=3,
        bottom=True,
        left=True,
        top=False,
        right=False
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    return fig, ax


def combine_plots_as_images_buffer(plots_list, nrows=1, ncols=None, figsize=(16, 6), title=None):
    """
    Объединяет графики с использованием буфера памяти (без временных файлов).
    """
    n_plots = len(plots_list)
    if ncols is None:
        ncols = n_plots

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, (fig_source, ax_source) in enumerate(plots_list):
        # Используем буфер
        buf = BytesIO()
        fig_source.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='white')
        buf.seek(0) 

        # Читаем изображение из буфера
        img = mpimg.imread(buf)
        axes[i].imshow(img)
        axes[i].axis('off')

        buf.close()

    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)

    plt.tight_layout()
    return fig, axes



def plot_two_distributions(series1: pd.Series, series2: pd.Series, 
                         label1: str = 'Series 1', 
                         label2: str = 'Series 2',
                         title: str = '',
                         xlabel: str = 'SNR',
                         ylabel: str = 'Dencity',
                         figsize: tuple = (6, 5),
                         alpha: float = 0.5):
    """
    Строит графики плотностей распределения
    """
    
    fig, ax = plt.subplots(figsize=figsize)  
    
  
    sns.kdeplot(data=series1, label=label1, alpha=alpha, linewidth=2, fill=True, ax=ax)
    sns.kdeplot(data=series2, label=label2, alpha=alpha, linewidth=2, fill=True, ax=ax)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Скрываем верхнюю и правую оси
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Делаем жирными левую и нижнюю оси (увеличиваем толщину линии)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()

    return fig, ax


