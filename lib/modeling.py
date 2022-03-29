from sklearn import pipeline, ensemble

from lib import data_processing


def build_model_pipeline(model_params={}) -> pipeline.Pipeline:
    return pipeline.Pipeline(
        steps=[("Modelling", ensemble.RandomForestRegressor(**model_params, n_jobs=-1))]
    )


def build_full_pipeline(
        max_features,
        ngram_min,
        ngram_max,
        numeric_col_names,
        categorical_col_names,
        geo_col_names,
        text_col_names,
        radiuses=(0.1, 1, 5),
) -> pipeline.Pipeline:
    # generate transformers
    numeric_transformer = data_processing.get_numeric_transformer()
    categoric_transformer = data_processing.get_categorical_transformer()
    geo_transformer = data_processing.get_geo_transformer(radiuses=radiuses)
    text_transformer = data_processing.get_text_transformer(
        max_features=max_features, ngram_min=ngram_min, ngram_max=ngram_max
    )

    # build featuring pipeline
    featuring_pipeline = data_processing.build_featuring_pipeline(
        numeric_transformer,
        categoric_transformer,
        geo_transformer,
        text_transformer,
        numeric_col_names,
        categorical_col_names,
        geo_col_names,
        text_col_names,
    )

    # build modeling pipeline
    model_pipeline = build_model_pipeline()

    # build full pipeline
    pipe = pipeline.Pipeline(
        steps=[("Featuring", featuring_pipeline), ("Modelling", model_pipeline)]
    )

    return pipe
