import pandas as pd

from .featureSelect import FeatureSelect_1D,RNA_seq
from .DeepFeature_1D import get_deep_feature
from .DeepFeature_2D import get_2DFeature


def model_get_feature(seq_fa):
    seq = RNA_seq(seq_fa)
    fea = FeatureSelect_1D(seq)
    feature_1D = fea.get_feature()

    [transformer_features, lstm_features, gru_features],pad_text = get_deep_feature(seq)
    deep_feature_2D = get_2DFeature(seq,pad_text)

    feature_transform = pd.concat([feature_1D, transformer_features, deep_feature_2D], axis=1)
    feature_lstm = pd.concat([feature_1D, lstm_features,deep_feature_2D], axis=1)
    feature_gru = pd.concat([feature_1D, gru_features,deep_feature_2D], axis=1)

    return feature_transform, feature_lstm, feature_gru
