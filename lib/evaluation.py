import numpy as np
import pandas as pd
from sklearn import pipeline
from matplotlib import pyplot as plt


def get_feature_names(
    pipe: pipeline.Pipeline,
    numeric_feature_names,
    categorical_feature_names,
    text_feature_names,
) -> np.array:
    """Get feature name from featuring pipeline."""
    res_numeric_feature_names = [x.upper() for x in numeric_feature_names]

    prep_transformers = pipe["Featuring"]["Preprocessing"].transformers_

    res_cat_feature_names = list(
        prep_transformers[1][1]["onehotencoder"].get_feature_names_out(
            [x.upper() for x in categorical_feature_names]
        )
    )

    res_geo_feature_names = prep_transformers[2][1]["geofeaturing"].get_feature_names_out()

    res_text_feature_names = []
    for j, fn in enumerate(text_feature_names):
        vec = prep_transformers[j + 3][1]["tfidfvectorizer"]
        new_fn = vec.get_feature_names_out()
        new_fn = [f"{fn.upper()}_{x}" for x in new_fn]
        res_text_feature_names.extend(new_fn)

    res_feature_names = np.array(
        res_numeric_feature_names
        + res_cat_feature_names
        + res_geo_feature_names
        + res_text_feature_names
    )

    return res_feature_names


def get_selected_feature_names(
    pipe: pipeline.Pipeline, input_feature_names
) -> np.array:
    """Get selected feature names from fitted featuring pipeline."""
    selected_feature_indexes = pipe["Featuring"]["Feature selection"].get_support()
    return input_feature_names[selected_feature_indexes]


def show_feature_importance(pipe, feature_names, n_top=20):
    """Visualize feature importance."""
    plt.figure(figsize=(8, 20))
    pd.DataFrame().assign(
        imp=pipe.steps[1][1][0].feature_importances_, feature=feature_names,
    ).set_index("feature").sort_values("imp").tail(n_top).plot.barh()
    plt.show()
