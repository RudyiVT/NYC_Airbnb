import re

import numpy as np
import pandas as pd
from sklearn import base, compose, ensemble, \
    feature_extraction, feature_selection, \
    neighbors, pipeline, preprocessing, impute

reshape_to_1d = preprocessing.FunctionTransformer(
    func=np.reshape, kw_args={"newshape": -1}
)
convert_to_series = preprocessing.FunctionTransformer(func=pd.Series)


def clean_text(txt: pd.Series) -> pd.Series:
    return (
        txt.astype(str).apply(
            lambda x: " ".join(re.findall("^[a-z]+|[A-Z][^A-Z]*", x))
        ).str.lower().str.replace(
            "[^\w\s]", " ", regex=True
        ).replace("\s+", " ", regex=True)
    )


class GeoFeaturing(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, radiuses=(0.1, 1, 3)):
        self.earth_radius = 6371
        assert len(radiuses) > 0, "Number of radiuses must be more then 0."
        self.radiuses = radiuses
        self.bts = {}
        self.prices = None
        self.feature_names = []

    def _get_count_in_radius(self, X, room_type, radius):
        inds, dists = self.bts.get(room_type).query_radius(
            np.radians(X),
            r=radius / self.earth_radius,
            return_distance=True,
        )

        filtered_inds = [i[d > 0] for i, d in zip(inds, dists)]
        return np.array([len(x) for x in filtered_inds])

    def _get_median_price_in_radius(self, X, room_type, radius):
        inds, dists = self.bts.get(room_type).query_radius(
            np.radians(X),
            r=radius / self.earth_radius,
            return_distance=True,
        )
        filtered_inds = [i[d > 0] for i, d in zip(inds, dists)]
        return np.array(
            [
                -1 if len(self.prices[i]) == 0 else np.median(self.prices[i])
                for i in filtered_inds
            ]
        )

    def fit(self, X, y=None):
        self.prices = X.iloc[:, 2].values
        self.room_types = X.iloc[:, 3].values
        coord_values = X.iloc[:, :2]
        for room_type in np.unique(self.room_types):
            filterd_coord_values = coord_values[room_type == self.room_types]
            self.bts[room_type] = neighbors.BallTree(
                np.radians(filterd_coord_values), metric="haversine"
            )
        return self

    def transform(self, X):
        counts = []
        median_prices = []
        coord_values = X.iloc[:, :2]
        self.feature_names = []
        for room_type in np.unique(self.room_types):
            for radius in self.radiuses:
                count_values = self._get_count_in_radius(
                    coord_values, room_type, radius
                )
                counts.append(count_values)

                median_values = self._get_median_price_in_radius(
                    coord_values, room_type, radius
                )
                median_prices.append(median_values)

                self.feature_names.extend(
                    [
                        f"count_{room_type}_radius_{radius}",
                        f"median_price_{room_type}_radius_{radius}",
                    ]
                )
        values = counts + median_prices
        return np.asarray(values).T

    def get_feature_names_out(self):
        return self.feature_names


def get_numeric_transformer():
    # process numeric features
    numeric_processor = preprocessing.FunctionTransformer(func=np.log1p)
    numeric_transformer = pipeline.make_pipeline(
        numeric_processor,
        impute.SimpleImputer(strategy="constant", fill_value=-1),
        verbose=True,
    )
    return numeric_transformer


def get_geo_transformer(radiuses=(0.1, 1, 5)):
    return pipeline.make_pipeline(GeoFeaturing(radiuses=radiuses))


def get_categorical_transformer():
    transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="constant", fill_value="UNKNOWN"),
        preprocessing.OneHotEncoder(handle_unknown="ignore"),
        verbose=True,
    )
    return transformer


def get_text_transformer(max_features=1000, ngram_min=1, ngram_max=3):
    # process text features
    text_processor = preprocessing.FunctionTransformer(func=clean_text)
    text_transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        convert_to_series,
        text_processor,
        feature_extraction.text.TfidfVectorizer(
            max_features=max_features, ngram_range=(ngram_min, ngram_max)
        ),
        verbose=True,
    )
    return text_transformer


def build_featuring_pipeline(
        numeric_transformer, categorical_transformer, geo_transformer, text_transformer,
        numeric_col_names, categorical_col_names, geo_col_names, text_col_names
) -> pipeline.Pipeline:
    feature_transformers = [
        ("Numeric", numeric_transformer, numeric_col_names),
        ("Categoric", categorical_transformer, categorical_col_names),
        ("Geo", geo_transformer, geo_col_names),
    ]

    for col_name in text_col_names:
        feature_transformers.append((f"Text: {col_name}", text_transformer, [col_name]))

    preprocessor_pipeline = compose.ColumnTransformer(
        feature_transformers, n_jobs=-1, remainder="drop", verbose=True
    )

    # build featuring pipeline
    featuring_pipeline = pipeline.Pipeline(
        steps=[
            ("Preprocessing", preprocessor_pipeline),
            (
                "Feature selection",
                feature_selection.SelectFromModel(
                    ensemble.RandomForestRegressor(n_jobs=-1), prefit=False
                ),
            ),
        ], verbose=True
    )

    return featuring_pipeline
