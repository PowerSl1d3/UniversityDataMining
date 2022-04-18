import numpy as np
import pandas as pd
from enum import Enum
import random
from copy import deepcopy
from itertools import combinations

random.seed(42)

enum_feature_possible_values = [chr(x + ord('a')) for x in range(26)]

global_alt = 0


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
        # задание 3
        self.possible_alternatives = None

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

    def generate_alternatives_for_concrete_medicine_history(self):
        flat_moments_of_observation = sum(self.concrete_moment_of_observation, [])
        alternatives = [[flat_moments_of_observation[-1] + 1]]
        for number_of_alternatives in range(2, 6):
            current_alternatives = list(
                filter(lambda x: x[-1] == flat_moments_of_observation[-1],
                       combinations(flat_moments_of_observation,
                                    number_of_alternatives)
                       )
            )
            alternatives_index_in_moments_of_observation = []
            for alternative in current_alternatives:
                alternatives_index_in_moments_of_observation.append(
                    list(
                        map(
                            lambda x: flat_moments_of_observation.index(x),
                            alternative
                        )
                    )
                )
            for alternative in alternatives_index_in_moments_of_observation:
                list_of_unique_alternatives = []
                is_first_alternative = True
                for range_start, range_end in zip(alternative[:-1], alternative[1:]):
                    if is_first_alternative:
                        list_of_unique_alternatives.append(
                            set(
                                self.concrete_values_for_periods_of_dynamic_of_observation_moment[:range_start + 1]
                            )
                        )
                        is_first_alternative = False
                    list_of_unique_alternatives.append(
                        set(
                            self.concrete_values_for_periods_of_dynamic_of_observation_moment[range_start + 1:range_end + 1]
                        )
                    )
                is_break = False
                for index in range(len(list_of_unique_alternatives) - 1):
                    if len(
                            list_of_unique_alternatives[index].intersection(list_of_unique_alternatives[index + 1])
                    ) > 0:
                        is_break = True
                if not is_break:
                    alternatives.append(
                        [flat_moments_of_observation[alternative_index] for alternative_index in alternative]
                    )
            self.possible_alternatives = alternatives

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
        for alternative in self.possible_alternatives:
            global global_alt
            global_alt += 1
            description += "Possible alternative with " + str(len(alternative)) + " are:\n"
            if len(alternative) == 1:
                description += f"[1, {alternative[0]})\n"
            else:
                for index in range(len(alternative) - 1):
                    if index == 0:
                        description += f"[1, {alternative[index] + 1})\n"
                    description += f"[{alternative[index] + 1}, {alternative[index + 1] + 1})\n"
        return description

    def __repr__(self):
        return self.__str__()


class Disease:
    def __init__(self, features_type: [FeatureType]):
        self.features = [Feature(x, index, None) for index, x in enumerate(features_type)]
        self.duration_of_period_dynamic = []
        self.writer = None

    def open_writer(self, filename: str):
        if self.writer is None:
            self.writer = pd.ExcelWriter(filename, engine="openpyxl")

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

        self.open_writer(filename)

        df_features.to_excel(self.writer, sheet_name="Sheet1", encoding="utf-8", startrow=0, startcol=3)
        df_possible_values.to_excel(self.writer, sheet_name="Sheet1", encoding="utf-8", startrow=0, startcol=6)
        df_normal_values.to_excel(self.writer, sheet_name="Sheet1", encoding="utf-8", startrow=0, startcol=9)
        df_clinical_picture.to_excel(self.writer, sheet_name="Sheet1", encoding="utf-8", startrow=0, startcol=12)
        df_number_of_periods_of_dynamic.to_excel(self.writer, sheet_name="Sheet1", encoding="utf-8", startrow=0, startcol=16)
        df_values_for_periods_of_dynamics.to_excel(self.writer, sheet_name="Sheet1", encoding="utf-8", startrow=0,
                                                   startcol=20)
        df_upper_and_down_time_bound.to_excel(self.writer, sheet_name="Sheet1", encoding="utf-8", startrow=0, startcol=26)

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

        self.open_writer(filename)

        df_medicine_history_without_concrete_values.to_excel(self.writer, sheet_name="Sheet2", encoding="utf-8", startrow=0, startcol=0)
        df_medicine_history_with_concrete_values.to_excel(self.writer, sheet_name="Sheet2", encoding="utf-8", startrow=0, startcol=7)

    def __str__(self):
        description = str(self.features)

        return description

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        if self.writer:
            self.writer.save()


def _make_disease() -> Disease:
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


def make_disease() -> Disease:
    disease = _make_disease()
    generate_limits_for_features(disease)
    generate_normal_values_for_features(disease)
    generate_number_of_periods_of_dynamic(disease)
    generate_values_for_periods_of_dynamic(disease)
    generate_upper_and_down_time_bound(disease)
    return disease


def make_medicine_history(disease: Disease):
    generate_duration_of_period_dynamic(disease)
    generate_number_of_observation_moments(disease)
    generate_concrete_medicine_history(disease)
    return disease


def generate_alternatives(disease: Disease) -> Disease:
    for feature in disease.features:
        feature.generate_alternatives_for_concrete_medicine_history()
    return disease


def main():
    assert (pd is not None)
    assert (np is not None)

    first_disease = make_disease()
    second_disease = deepcopy(first_disease)
    make_medicine_history(first_disease)
    make_medicine_history(second_disease)

    generate_alternatives(first_disease)

    print(first_disease)

    generate_alternatives(second_disease)

    print(second_disease)

    #disease.make_report_about_disease("tmp.xls")
    #disease.make_report_about_medicine_history("tmp.xls")


if __name__ == "__main__":
    main()
    print("ALTERNATIVES TOTAL:", global_alt)

# FeatureType.BOOL with possible values: [True, False] and normal values: True and number of periods of dynamic: 5
# Values for periods of dynamics: [True, False, True, False, True]
# Values for upper and down time bound of dynamics: [(7, 13), (9, 14), (5, 17), (8, 17), (5, 20)]
# Time duration for concrete period dynamic: [11, 14, 7, 16, 12]
# Number of observable moment for concrete period dynamic: [3, 2, 1, 3, 2]
# Concrete moments of observation moment: [[3, 4, 6], [19, 20], [26], [34, 46, 48], [51, 59]]
# Concrete values at moments of observation: [True, False, True, False, False, True, False, False, True, False, False]
# Possible alternative with 1 are:
# [1, 60)
