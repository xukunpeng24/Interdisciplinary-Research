import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle

def load_config(file_path):
    with open(file_path) as f:
        return json.load(f)


def calculate_novelty_means(df, subject_categories, novelty_columns):
    
    # 准备存储不同类型科目的DataFrame和平均新颖性分数
    dfs = {}
    novelty_means = {}

    # 遍历每个科目类别，过滤DataFrame并计算平均新颖性分数
    for category_name, subjects in subject_categories.items():
        filtered_df = df[df['group_subject_mapped'].apply(lambda x: any(subject in subjects for subject in x))]
        dfs[category_name] = filtered_df
        
        avg_novelty = filtered_df[novelty_columns].mean().round(2)
        novelty_means[category_name] = tuple(avg_novelty.values)
    
    return novelty_means



def plot_novelty(novelty_means, novelty_columns):
    colormap = plt.get_cmap('Blues')
    colors = colormap(np.linspace(0.5, 1, len(novelty_means)))

    # Extracting data for the plots
    categories_data = {
        category: [novelty_means[key][i] for key in novelty_means]
        for i, category in enumerate(novelty_columns)
    }

    # Plotting
    fig, axs = plt.subplots(1, len(novelty_columns), figsize=(15, 5))
    
    for ax, (category, data) in zip(axs, categories_data.items()):
        ax.bar(novelty_means.keys(), data, color=colors)
        ax.set_title(category)
        ax.set_ylabel(f'{category} scores')
        ax.set_xlabel('Types')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':


    subject_config = load_config(r'config\subjects.json')
    # color_config = load_config('config/colors.json')

    subject_index = subject_config['subject_index']
    natural_sciences = subject_config['natural_sciences']
    social_sciences = subject_config['social_sciences']
    multiple_sciences = subject_config['multiple_sciences']
    # colors = color_config['colors']

    subject_categories = {
    'Natural': natural_sciences,
    'Social': social_sciences,
    'Multi': multiple_sciences,
    'Natural-Social': natural_sciences + social_sciences  # Assuming this is the intended logic for Natural-Social
}

    novelty_columns = ['career_novelty', 'team_novelty', 'expedition_novelty']

    with open(r'data\processed\novelty+group.pkl', 'rb') as file:
        df = pickle.load(file)
    df['group_subject_mapped'] = df['group_subject'].apply(lambda groups: [subject_index[str(group)] for group in groups if str(group) in subject_index])

    novelty_means = calculate_novelty_means(df, subject_categories, novelty_columns)


    plot_novelty(novelty_means, novelty_columns)