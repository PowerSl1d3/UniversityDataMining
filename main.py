import numpy as np
import pandas as pd
from enum import Enum
import random

random.seed(42)

enum_feature_possible_values = [chr(x + ord('a')) for x in range(26)]


def prefix_sum(array: [int]) -> [int]:
    answer = [0]
    for value in array:
        answer.append(answer[-1] + value)
    return answer


class FeatureType(Enum):
    BOOL = 0
    ENUM = 1
    INTEGRAL = 2


class Feature:
    def __init__(self, feature_type: FeatureType, feature_index: int, feature_bound):
        # задание 1
        self.title = f"Признак{feature_index}"
        self.type = feature_type
        # для BOOL типов ограничения по сути не нужны
        # для  ENUM типов ограничением будет количество этих самых вариантов значений
        # для INTEGRAL типов ограничениями будет диапазон целых значений
        self.bound = feature_bound
        self.normal_value = None
        self.number_of_periods_of_dynamic = None
        self.values_for_periods_of_dynamic = []
        self.upper_and_down_time_bound = []
        # задание 2
        self.duration_of_period_dynamic = []
        self.number_of_observation_moments = []
        self.concrete_moment_of_observation = []
        self.concrete_values_for_periods_of_dynamic_of_observation_moment = []

    def get_possible_values_representation(self) -> str:
        if self.type is FeatureType.BOOL:
            return "[есть, нет]"
        else:
            return str(self.bound)

    def get_normal_value_representation(self) -> str:
        if self.type is FeatureType.BOOL:
            return "есть" if self.normal_value is True else "нет"
        else:
            return str(self.normal_value)

    def _get_values_for_periods_of_dynamic_representation(self, index: int) -> str:
        if self.type is FeatureType.BOOL:
            return "есть" if self.values_for_periods_of_dynamic[index] is True else "нет"
        else:
            return self.values_for_periods_of_dynamic[index]

    def generate_values_for_periods_of_dynamic(self):
        self.values_for_periods_of_dynamic = [None] * self.number_of_periods_of_dynamic
        last_appended_value = None
        for index in range(self.number_of_periods_of_dynamic):
            if self.type is FeatureType.BOOL:
                generated_item = self.bound[random.randint(0, 1)]
                if generated_item != last_appended_value:
                    self.values_for_periods_of_dynamic[index] = generated_item
                else:
                    self.values_for_periods_of_dynamic[index] = not generated_item
            elif self.type is FeatureType.ENUM:
                generated_item = self.bound[random.randint(0, len(self.bound) - 1)]
                while generated_item == last_appended_value:
                    generated_item = self.bound[random.randint(0, len(self.bound) - 1)]
                self.values_for_periods_of_dynamic[index] = generated_item
            else:
                generated_item = random.randint(0, self.bound[-1])
                while generated_item == last_appended_value:
                    generated_item = random.randint(0, self.bound[-1])
                self.values_for_periods_of_dynamic[index] = generated_item
            last_appended_value = self.values_for_periods_of_dynamic[index]

    def generate_upper_and_down_time_bound(self):
        for _ in range(self.number_of_periods_of_dynamic):
            lower_bound = random.randint(5, 10)
            upper_bound = random.randint(lower_bound + 1, 20)
            self.upper_and_down_time_bound.append((lower_bound, upper_bound))

    def generate_duration_of_period_dynamic(self):
        self.duration_of_period_dynamic = [random.randint(
            self.upper_and_down_time_bound[index][0],
            self.upper_and_down_time_bound[index][1]
        ) for index in range(self.number_of_periods_of_dynamic)]

    def generate_number_of_observation_moments(self):
        current_number_of_observable_moments = []
        for i in range(self.number_of_periods_of_dynamic):
            current_number_of_observable_moments.append(random.randint(1, 3))
            if current_number_of_observable_moments[-1] > self.duration_of_period_dynamic[i]:
                current_number_of_observable_moments[-1] = self.duration_of_period_dynamic[i]
        self.number_of_observation_moments = current_number_of_observable_moments

    def generate_concrete_medicine_history(self):
        self.concrete_moment_of_observation = []
        self.concrete_values_for_periods_of_dynamic_of_observation_moment = []
        ps_period_dynamic = prefix_sum(self.duration_of_period_dynamic)
        for period_index, time_interval in enumerate(
                zip(ps_period_dynamic[:-1], ps_period_dynamic[1:])
        ):
            self.concrete_moment_of_observation.append(sorted(random.sample(range(
                time_interval[0] + 1,
                time_interval[1] + 1),
                self.number_of_observation_moments[period_index]
            )))

        skip_one = False
        for index, moment_of_observation in enumerate(self.concrete_moment_of_observation):
            skip_one = index != 0
            if skip_one:
                appended_value = self.concrete_values_for_periods_of_dynamic_of_observation_moment[-1]
                while appended_value == self.concrete_values_for_periods_of_dynamic_of_observation_moment[-1]:
                    appended_value = random.choice(self.bound) if self.type is not FeatureType.INTEGRAL else random.randint(self.bound[0], self.bound[1])
                self.concrete_values_for_periods_of_dynamic_of_observation_moment.append(appended_value)
            for time_moment in moment_of_observation[0 + int(skip_one):]:
                self.concrete_values_for_periods_of_dynamic_of_observation_moment.append(
                    random.choice(self.bound) if self.type is not FeatureType.INTEGRAL else random.randint(self.bound[0], self.bound[1])
                )

    def __str__(self):
        description = str()
        description += f"{self.type} with possible values: {self.bound}"
        description += f" and normal values: {self.normal_value}"
        description += f" and number of periods of dynamic: {self.number_of_periods_of_dynamic}"
        description += "\n"
        description += f"Values for periods of dynamics: {self.values_for_periods_of_dynamic}"
        description += "\n"
        description += f"Values for upper and down time bound of dynamics: {self.upper_and_down_time_bound}"
        description += "\n"
        description += f"Time duration for concrete period dynamic: {self.duration_of_period_dynamic}"
        description += "\n"
        description += f"Number of observable moment for concrete period dynamic: {self.number_of_observation_moments}"
        description += "\n"
        description += f"Concrete moments of observation moment: {self.concrete_moment_of_observation}"
        description += "\n"
        description += f"Concrete values at moments of observation: {self.concrete_values_for_periods_of_dynamic_of_observation_moment}"
        description += "\n"
        return description

    def __repr__(self):
        return self.__str__()


class Disease:
    def __init__(self, features_type: [FeatureType]):
        self.features = [Feature(x, index, None) for index, x in enumerate(features_type)]
        self.duration_of_period_dynamic = []
        self.number_of_observation_moments = []
        self.writer = None

    def _open_writer(self, filename):
        self.writer = open(filename, mode="w")

    def make_report_about_disease(self, filename: str):
        df_features = pd.DataFrame(data=[f"Признак{index}" for index in range(len(self.features))],
                                   columns=["Признаки"])
        df_possible_values = pd.DataFrame(
            data=[feature.get_possible_values_representation() for feature in self.features],
            index=[feature.title for feature in self.features],
            columns=["Возможные значения"])

        df_normal_values = pd.DataFrame(data=[feature.get_normal_value_representation() for feature in self.features],
                                        index=[feature.title for feature in self.features],
                                        columns=["Нормальные значения"])

        # TODO: сделать интексацию для заболеваний, пропихнув параметр извне
        df_clinical_picture = pd.DataFrame(data=[feature.title for feature in self.features],
                                           index=[self.features[0].title for _ in range(len(self.features))],
                                           columns=["Клиническая картина"])

        df_number_of_periods_of_dynamic = pd.DataFrame(data=[("Заболевание0",
                                                              feature.title,
                                                              feature.number_of_periods_of_dynamic)
                                                             for feature in self.features]) \
            .pivot_table(index=[0, 1], aggfunc='first')

        df_values_for_periods_of_dynamics = []
        for feature in self.features:
            for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
                df_values_for_periods_of_dynamics.append(
                    (
                        "Заболевание0",
                        feature.title,
                        feature.number_of_periods_of_dynamic,
                        current_number_of_period_dynamic,
                        feature._get_values_for_periods_of_dynamic_representation(
                            index=current_number_of_period_dynamic)
                    )
                )

        df_values_for_periods_of_dynamics = pd.DataFrame(data=df_values_for_periods_of_dynamics).pivot_table(
            index=[0, 1, 2, 3],
            aggfunc='first'
        )

        df_upper_and_down_time_bound = []
        for feature in self.features:
            for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
                df_upper_and_down_time_bound.append(
                    (
                        "Заболевание0",
                        feature.title,
                        feature.number_of_periods_of_dynamic,
                        current_number_of_period_dynamic,
                        feature.upper_and_down_time_bound[current_number_of_period_dynamic][0],
                        feature.upper_and_down_time_bound[current_number_of_period_dynamic][1]
                    )
                )

        df_upper_and_down_time_bound = pd.DataFrame(data=df_upper_and_down_time_bound).pivot_table(
            index=[0, 1, 2, 3],
            aggfunc='first'
        )

        if self.writer is None:
            self._open_writer(filename)
        df_features.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")
        df_possible_values.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")
        df_normal_values.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")
        df_clinical_picture.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")
        df_number_of_periods_of_dynamic.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")
        df_values_for_periods_of_dynamics.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")
        df_upper_and_down_time_bound.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")

    def make_report_about_medicine_history(self, filename: str):
        df_medicine_history_without_concrete_values = []
        for feature in self.features:
            for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
                df_medicine_history_without_concrete_values.append(
                    (
                        "ИБ0",
                        "Заболевание0",
                        feature.title,
                        current_number_of_period_dynamic,
                        feature.duration_of_period_dynamic[current_number_of_period_dynamic],
                        feature.number_of_observation_moments[current_number_of_period_dynamic]
                    )
                )
        df_medicine_history_without_concrete_values = pd.DataFrame(data=df_medicine_history_without_concrete_values).pivot_table(
            index=[0, 1, 2, 3],
            aggfunc="first"
        )

        df_medicine_history_with_concrete_values = []
        for feature in self.features:
            tmp_concrete_time_values = sum(feature.concrete_moment_of_observation, [])
            for current_number_of_period_dynamic in range(len(tmp_concrete_time_values)):
                df_medicine_history_with_concrete_values.append(
                    (
                        "ИБ0",
                        "Заболевание0",
                        feature.title,
                        tmp_concrete_time_values[current_number_of_period_dynamic],
                        feature.concrete_values_for_periods_of_dynamic_of_observation_moment[current_number_of_period_dynamic]
                    )
                )

        df_medicine_history_with_concrete_values = pd.DataFrame(data=df_medicine_history_with_concrete_values).pivot_table(
            index=[0, 1, 2, 3],
            aggfunc="first"
        )

        if self.writer is None:
            self._open_writer(filename)
        df_medicine_history_without_concrete_values.to_csv(self.writer, encoding="utf-8")
        self.writer.write("\n")
        df_medicine_history_with_concrete_values.to_csv(self.writer, encoding="utf-8")

    def __str__(self):
        description = str(self.features)

        return description

    def __repr__(self):
        return self.__str__()


def make_disease() -> Disease:
    return Disease([
        FeatureType.BOOL,
        FeatureType.INTEGRAL,
        FeatureType.INTEGRAL,
        FeatureType.ENUM,
        FeatureType.ENUM,
        FeatureType.ENUM
    ])


def generate_limits_for_features(disease: Disease) -> Disease:
    for feature in disease.features:
        if feature.type is FeatureType.BOOL:
            feature.bound = [True, False]
        elif feature.type is FeatureType.ENUM:
            feature.bound = enum_feature_possible_values[:random.randint(3, 5)]
        else:
            feature.bound = [0, random.randint(100, 200)]
    return disease


def generate_normal_values_for_features(disease: Disease) -> Disease:
    for feature in disease.features:
        if feature.type is FeatureType.BOOL:
            feature.normal_value = feature.bound[random.randint(0, 1)]
        elif feature.type is FeatureType.ENUM:
            feature.normal_value = feature.bound[random.randint(0, len(feature.bound) - 1)]
        else:
            feature.normal_value = [0, random.randrange(feature.bound[0], feature.bound[-1] // 2)]
    return disease


def generate_number_of_periods_of_dynamic(disease: Disease) -> Disease:
    for feature in disease.features:
        feature.number_of_periods_of_dynamic = random.randint(3, 5)
    return disease


def generate_values_for_periods_of_dynamic(disease: Disease) -> Disease:
    for feature in disease.features:
        feature.generate_values_for_periods_of_dynamic()
    return disease


def generate_upper_and_down_time_bound(disease: Disease) -> Disease:
    for feature in disease.features:
        feature.generate_upper_and_down_time_bound()
    return disease


def generate_duration_of_period_dynamic(disease: Disease) -> Disease:
    for feature in disease.features:
        feature.generate_duration_of_period_dynamic()
    return disease


def generate_number_of_observation_moments(disease: Disease) -> Disease:
    for feature in disease.features:
        feature.generate_number_of_observation_moments()
    return disease


def generate_concrete_medicine_history(disease: Disease) -> Disease:
    for feature in disease.features:
        feature.generate_concrete_medicine_history()
    return disease


def main():
    assert (pd is not None)
    assert (np is not None)

    disease = make_disease()
    generate_limits_for_features(disease)
    generate_normal_values_for_features(disease)
    generate_number_of_periods_of_dynamic(disease)
    generate_values_for_periods_of_dynamic(disease)
    generate_upper_and_down_time_bound(disease)
    generate_duration_of_period_dynamic(disease)
    generate_number_of_observation_moments(disease)
    generate_concrete_medicine_history(disease)

    print(disease, sep="\n")

    disease.make_report_about_disease("tmp.csv")
    disease.make_report_about_medicine_history("tmp.csv")


if __name__ == "__main__":
    main()
