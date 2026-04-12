import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import math


def plot_snr_histogram(data1, data2=None, bins=20, xlim=None, ylim=None, 
                                  title="SNR Distribution", 
                                  xlabel="SNR dB in data", ylabel="Count",
                                  legend= False,
                                  label1="label1", label2="label2",
                                  figsize=(10, 6), filepath=None):
    """
    Построение наложенных гистограмм для двух тензоров (подвыборок).
    """
    
    # Настройка стиля
    plt.figure(figsize=figsize)
    sns.set_style("white")
    
    # Построение двух гистограмм с наложением
    sns.kdeplot(data=data1, label=label1, alpha=0.3, linewidth=2, fill=True)
    s1_mean = data1.mean()
    plt.axvline(x=s1_mean, color='blue', linewidth=1, linestyle='--', alpha=0.8)
    plt.annotate(
        f'{s1_mean:.2f}',
        xy=(s1_mean, 1),  
        xytext=(10, 5), 
        textcoords='offset points',
        fontsize=10,
        color='blue',
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
    )

    if data2 is not None:
        sns.kdeplot(data=data2, label=label2, alpha=0.3, linewidth=2, fill=True)
        s2_mean = data2.mean()
        plt.axvline(x=s2_mean, color='red', linewidth=1, linestyle='--', alpha=0.8)
        plt.annotate(
            f'{s2_mean:.2f}',
            xy=(s2_mean, 2),  
            xytext=(10, 5), 
            textcoords='offset points',
            fontsize=10,
            color='red',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
        )
    
    # Настройка осей и заголовка
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Установка границ осей, если переданы
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # Настройка границ: жирные нижняя и левая, остальные — отсутствуют
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Удаляем метки на верхней и правой осях
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if legend:
        # Добавляем легенду
        plt.legend(fontsize=10, frameon=True, framealpha=0.8,)
        #plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')  # под графиком по центру
    
    # Добавляем горизонтальный grid (неброский)
    ax.yaxis.grid(True, linestyle=':', alpha=0.3, linewidth=0.8)
    
    # Сохраняем график, если указан путь
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    else:
        # Отображение графика
        plt.tight_layout()
        plt.show()


def plot_snr_histogram_old(data1, data2=None, bins=20, xlim=None, ylim=None, 
                                  title="SNR Distribution", 
                                  xlabel="SNR dB in data", ylabel="Count",
                                  legend= False,
                                  label1="label1", label2="label2",
                                  figsize=(10, 6), filepath=None):
    """
    Построение наложенных гистограмм для двух тензоров (подвыборок).
    """
    
    # Настройка стиля
    plt.figure(figsize=figsize)
    sns.set_style("white")
    
    # Построение двух гистограмм с наложением
    plt.hist(data1, bins=bins, color='#D3D3D3', alpha=0.6, edgecolor='black', linewidth=0.8, label=label1)

    if data2 is not None:
        plt.hist(data2, bins=bins, color='#1F77B4', alpha=0.3, edgecolor='black', linewidth=0.8, label=label2)
    
    # Настройка осей и заголовка
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Установка границ осей, если переданы
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # Настройка границ: жирные нижняя и левая, остальные — отсутствуют
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Удаляем метки на верхней и правой осях
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if legend:
        # Добавляем легенду
        plt.legend(fontsize=10, frameon=True, framealpha=0.8,)
        #plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')  # под графиком по центру
    
    # Добавляем горизонтальный grid (неброский)
    ax.yaxis.grid(True, linestyle=':', alpha=0.3, linewidth=0.8)
    
    # Сохраняем график, если указан путь
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    else:
        # Отображение графика
        plt.tight_layout()
        plt.show()

def plot_snr_distribution_comparison(df: pd.DataFrame) -> None:
    """
    Функция строит сетку гистограмм для сравнения распределения значений (SNR) 
    между полным зашумленным датасетом и различными отфильтрованными подмножествами.
    """

    # --- HYPERPARAMETERS -----------------------------
    BASELINE_NAME = 'H_noisy'           # Имя датасета, с которым идет сравнение
    COLOR_BASELINE = '#bbbbbb'          # Цвет базового датасета (серый)
    COLOR_SUBSET = '#1f77b4'            # Цвет сравниваемого подмножества (синий)
    
    N_COLS = 3                          # Количество колонок в сетке графиков
    FIG_WIDTH = 10                      # Ширина всей фигуры
    FIG_HEIGHT_PER_ROW = 4              # Высота одной строки графиков
    
    X_LABEL = "SNR dB in data"          # Подпись оси X
    Y_LABEL = "Count"                   # Подпись оси Y
    TITLE_FONT_SIZE = 14                # Размер шрифта заголовков
    AXIS_LABEL_FONT_SIZE = 12           # Размер шрифта подписей осей
    
    LEGEND_LABEL_FULL = 'Full Data (Noisy)' # Текст легенды для базового набора
    LEGEND_LABEL_SUB = 'Retained Data'      # Текст легенды для подмножества
    # -------------------------------------------------

    # 1. Подготовка списка подмножеств для итерации (уникальные по 'dataset')
    subset_names = [name for name in df['dataset'].unique() if name != BASELINE_NAME]
    n_plots = len(subset_names)

    if n_plots == 0:
        print("Warning: No subsets found to compare against baseline.")
        return None

    # 2. Расчет размеров сетки
    n_rows = math.ceil(n_plots / N_COLS)
    
    # Динамический расчет высоты фигуры в зависимости от количества строк
    total_height = n_rows * FIG_HEIGHT_PER_ROW

    # 3. Настройка стиля и создание фигуры
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(
        n_rows, 
        N_COLS, 
        figsize=(FIG_WIDTH, total_height), 
        sharey=False, 
        sharex=False
    )
    
    # Приводим axes к плоскому массиву для удобной итерации (даже если 1 строка)
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    # 4. Основной цикл построения графиков
    for i, sub_name in enumerate(subset_names):
        ax = axes_flat[i]
        
        # Создаем временный DF с двумя нужными категориями для текущего графика (для hue)
        current_mask = df['dataset'].isin([BASELINE_NAME, sub_name])
        current_view_df = df[current_mask]
        
        sns.histplot(
            data=current_view_df,
            x="value",
            hue="dataset",
            hue_order=[BASELINE_NAME, sub_name], # Гарантирует порядок цветов
            stat="count",
            discrete=True,
            multiple="dodge",     # Столбцы рядом друг с другом
            shrink=0.8,           # Ширина столбцов (отступ друг от друга)
            palette={BASELINE_NAME: COLOR_BASELINE, sub_name: COLOR_SUBSET},
            edgecolor=None,
            legend=False,         # Отключаем локальную легенду на каждом графике
            ax=ax
        )
        
        # Оформление графика
        ax.set_title(f"{sub_name}", fontsize=TITLE_FONT_SIZE, weight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
        
        # Убираем рамки сверху и справа
        sns.despine(ax=ax)

    # 5. Очистка пустых областей сетки (если графиков меньше, чем ячеек)
    for j in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # 6. Формирование глобальной легенды
    full_patch = mpatches.Patch(color=COLOR_BASELINE, label=LEGEND_LABEL_FULL)
    sub_patch = mpatches.Patch(color=COLOR_SUBSET, label=LEGEND_LABEL_SUB)

    fig.legend(
        handles=[full_patch, sub_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02), # Немного выше заголовка
        ncol=2,
        frameon=False,
        fontsize=AXIS_LABEL_FONT_SIZE
    )

    
    plt.tight_layout()
    plt.show()

    # return - ничего не возвращаем