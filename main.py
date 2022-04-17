import numpy as np
import pandas as pd
from enum import Enum
import random

random.seed(42)

enum_feature_possible_values = [chr(x + ord('a')) for x in range(26)]


class FeatureType(Enum):
    BOOL = 0
    ENUM = 1
    INTEGRAL = 2


class Feature:
    def __init__(self, feature_type: FeatureType, feature_bound):
        self.type = feature_type
        # для BOOL типов ограничения по сути не нужны
        # для  ENUM типов ограничением будет количество этих самых вариантов значений
        # для INTEGRAL типов ограничениями будет диапазон целых значений
        self.bound = feature_bound
        self.normal_value = None

    def __str__(self):
        return str(f"{self.type} with possible values: {self.bound} and normal values: {self.normal_value}\n")

    def __repr__(self):
        return self.__str__()


class Disease:
    def __init__(self, features_type: [FeatureType]):
        self.features = [Feature(x, None) for x in features_type]

    def __str__(self):
        return str(self.features)

    def __repr__(self):
        return str(self.features)


def make_disease() -> Disease:
    return Disease([
        FeatureType.BOOL,
        FeatureType.INTEGRAL,
        FeatureType.INTEGRAL,
        FeatureType.ENUM,
        FeatureType.ENUM,
        FeatureType.ENUM,
        FeatureType.ENUM
    ])


def set_limits_for_features(disease: Disease) -> Disease:
    for feature in disease.features:
        if feature.type is FeatureType.BOOL:
            feature.bound = [True, False]
        elif feature.type is FeatureType.ENUM:
            feature.bound = enum_feature_possible_values[:random.randint(3, 5)]
        else:
            feature.bound = range(0, random.randint(100, 200))
    return disease


def set_normal_values_for_features(disease: Disease) -> Disease:
    for feature in disease.features:
        if feature.type is FeatureType.BOOL:
            feature.normal_value = feature.bound[random.randint(0, 1)]
        elif feature.type is FeatureType.ENUM:
            feature.normal_value = feature.bound[random.randint(0, len(feature.bound) - 1)]
        else:
            feature.normal_value = random.randrange(feature.bound.start, feature.bound.stop)
    return disease


def main():
    assert (pd is not None)
    assert (np is not None)

    disease = make_disease()
    disease = set_limits_for_features(disease)
    disease = set_normal_values_for_features(disease)
    print(disease, sep="\n")


if __name__ == "__main__":
    main()
