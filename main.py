import itertools

import numpy as np
import pandas as pd
from enum import Enum
import random
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as Soup

random.seed(13)
plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
writer = pd.ExcelWriter("ИАД.Б9118-09.03.04прогин.АксененкоОлег.xls", engine="openpyxl")
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
        self.good_alternatives = {}
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

    def get_possible_alternative_representation(self, index: int) -> str:
        current_alternative = self.possible_alternatives[index]
        description = str()
        if len(current_alternative) == 1:
            return f"[1, {current_alternative[0] + 1})"
        description += f"[1, {current_alternative[0] + 1})"
        for index in range(len(current_alternative) - 1):
            description += f" -> [{current_alternative[index] + 1}, {current_alternative[index + 1] + 1})"
        return description

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
            # Тут задаётся количество значений для периода
            current_number_of_observable_moments.append(random.randint(3, 5))
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
                    appended_value = random.choice(
                        self.bound) if self.type is not FeatureType.INTEGRAL else random.randint(self.bound[0],
                                                                                                 self.bound[1])
                self.concrete_values_for_periods_of_dynamic_of_observation_moment.append(appended_value)
            for _ in moment_of_observation[0 + int(skip_one):]:
                self.concrete_values_for_periods_of_dynamic_of_observation_moment.append(
                    random.choice(self.bound) if self.type is not FeatureType.INTEGRAL else random.randint(
                        self.bound[0], self.bound[1])
                )

    def generate_alternatives_for_concrete_medicine_history(self):
        flat_moments_of_observation = sum(self.concrete_moment_of_observation, [])
        alternatives = [[flat_moments_of_observation[-1]]]
        self.good_alternatives[len(flat_moments_of_observation) - 1,] = [set(
            self.concrete_values_for_periods_of_dynamic_of_observation_moment
        )]
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
                            self.concrete_values_for_periods_of_dynamic_of_observation_moment[
                            range_start + 1:range_end + 1]
                        )
                    )
                is_break = False
                for index in range(len(list_of_unique_alternatives) - 1):
                    if len(
                            list_of_unique_alternatives[index].intersection(list_of_unique_alternatives[index + 1])
                    ) > 0:
                        is_break = True
                if not is_break:
                    self.good_alternatives[tuple(alternative)] = list_of_unique_alternatives
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
                description += f"[1, {alternative[0] + 1})\n"
            else:
                for index in range(len(alternative) - 1):
                    if index == 0:
                        description += f"[1, {alternative[index] + 1})\n"
                    description += f"[{alternative[index] + 1}, {alternative[index + 1] + 1})\n"
        return description

    def __repr__(self):
        return self.__str__()


class Disease:
    def __init__(self, features_type: [FeatureType], title: str):
        self.title = title
        self.medicine_history_title = None
        self.features = [Feature(x, index, None) for index, x in enumerate(features_type)]

    def set_medicine_history_title(self, title: str):
        self.medicine_history_title = title

    def __str__(self):
        description = str(self.features)

        return description

    def __repr__(self):
        return self.__str__()


def _make_disease() -> Disease:
    return Disease([
        FeatureType.BOOL,
        FeatureType.INTEGRAL,
        FeatureType.ENUM,
        FeatureType.ENUM,
        FeatureType.ENUM,
        FeatureType.ENUM
    ], title="Заболевание0")


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


def make_possible_alternatives_description(feature: Feature) -> str:
    description = str()
    for alternative in feature.possible_alternatives:
        description += "Possible alternative with " + str(len(alternative)) + " are:\n"
        if len(alternative) == 1:
            description += f"[1, {alternative[0] + 1})\n"
        else:
            for index in range(len(alternative) - 1):
                if index == 0:
                    description += f"[1, {alternative[index] + 1})\n"
                description += f"[{alternative[index] + 1}, {alternative[index + 1] + 1})\n"


def reduce_alternatives_for_medicine_story(diseases: [Disease]) -> []:
    good_feature_alternatives = []
    bad_feature_alternatives = []
    all_disease_combinations = list(combinations(diseases, 2))
    for number_of_dynamic_periods in range(2, 6):
        for first_disease, second_disease in tqdm(all_disease_combinations):
            for first_feature, second_feature in zip(first_disease.features, second_disease.features):
                # TODO: delete
                if first_feature.type is not FeatureType.ENUM:
                    continue
                alternatives_with_target_periods_of_dynamic_of_first_feature = list(filter(
                    lambda x: len(x) == number_of_dynamic_periods,
                    first_feature.possible_alternatives
                ))
                alternatives_with_target_periods_of_dynamic_of_second_feature = list(filter(
                    lambda x: len(x) == number_of_dynamic_periods,
                    second_feature.possible_alternatives
                ))
                if (len(alternatives_with_target_periods_of_dynamic_of_first_feature) == 0 or
                        len(alternatives_with_target_periods_of_dynamic_of_second_feature) == 0):
                    continue
                first_alternatives_index_in_moments_of_observation = []
                first_flat_moments_of_observation = sum(first_feature.concrete_moment_of_observation, [])
                for alternative in alternatives_with_target_periods_of_dynamic_of_first_feature:
                    first_alternatives_index_in_moments_of_observation.append(
                        list(
                            map(
                                lambda x: first_flat_moments_of_observation.index(x),
                                alternative
                            )
                        )
                    )
                second_alternatives_index_in_moments_of_observation = []
                second_flat_moments_of_observation = sum(second_feature.concrete_moment_of_observation, [])
                for alternative in alternatives_with_target_periods_of_dynamic_of_second_feature:
                    second_alternatives_index_in_moments_of_observation.append(
                        list(
                            map(
                                lambda x: second_flat_moments_of_observation.index(x),
                                alternative
                            )
                        )
                    )
                for first_alternative in first_alternatives_index_in_moments_of_observation:
                    for second_alternative in second_alternatives_index_in_moments_of_observation:
                        current_table = []
                        for dynamic_period_index in range(len(first_alternative)):
                            current_table.append((
                                first_disease.medicine_history_title + " —> " + second_disease.medicine_history_title,
                                ("Альтернатива " + str(
                                    first_alternatives_index_in_moments_of_observation.index(first_alternative)) +
                                 " сравнивается с альтернативой " + str(
                                            second_alternatives_index_in_moments_of_observation.index(
                                                second_alternative))),
                                dynamic_period_index,
                                first_feature.good_alternatives[tuple(first_alternative)][dynamic_period_index].union(
                                    second_feature.good_alternatives[tuple(second_alternative)][dynamic_period_index]),
                                (
                                    min((
                                            first_flat_moments_of_observation[first_alternative[dynamic_period_index]]
                                            -
                                            (0 if dynamic_period_index == 0 else first_flat_moments_of_observation[
                                                first_alternative[dynamic_period_index - 1]])
                                    ),
                                        (
                                                second_flat_moments_of_observation[
                                                    second_alternative[dynamic_period_index]]
                                                -
                                                (0 if dynamic_period_index == 0 else second_flat_moments_of_observation[
                                                    second_alternative[dynamic_period_index - 1]])
                                        )
                                    ),
                                    max((
                                            first_flat_moments_of_observation[first_alternative[dynamic_period_index]]
                                            -
                                            (0 if dynamic_period_index == 0 else first_flat_moments_of_observation[
                                                first_alternative[dynamic_period_index - 1]])
                                    ),
                                        (
                                                second_flat_moments_of_observation[
                                                    second_alternative[dynamic_period_index]]
                                                -
                                                (0 if dynamic_period_index == 0 else second_flat_moments_of_observation[
                                                    second_alternative[dynamic_period_index - 1]])
                                        ))
                                )
                            ))
                        is_break = False
                        for index in range(len(current_table) - 1):
                            if len(current_table[index][3].intersection(current_table[index + 1][3])) != 0:
                                # print("BAD TABLE")
                                # print(pd.DataFrame(data=current_table).to_markdown(index=False))
                                # print("-" * 30)
                                is_break = True
                                break
                        if not is_break:
                            good_feature_alternatives.append(current_table)
                        else:
                            bad_feature_alternatives.append(current_table)
    # for element in good_feature_alternatives:
    #     element = pd.DataFrame(data=element)
    #     print(element.to_markdown(index=False))
    #     print("-" * 30)
    # print(len(good_feature_alternatives))
    return good_feature_alternatives, bad_feature_alternatives


def make_first_report(diseases: [Disease]) -> None:
    df_disease = pd.DataFrame(data=[disease.title for disease in diseases], columns=["Заболевания"])

    df_disease.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=0, startcol=0, index=False)

    df_features = pd.DataFrame(data=[f"Признак{index}" for index in range(len(diseases[0].features))],
                               columns=["Признаки"])

    df_features.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=0, startcol=2, index=False)

    df_possible_values = pd.DataFrame(
        data=[feature.get_possible_values_representation() for feature in diseases[0].features],
        index=[feature.title for feature in diseases[0].features],
        columns=["Возможные значения (ВЗ)"])

    df_possible_values.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=0, startcol=4, index=True)

    df_normal_values = pd.DataFrame(
        data=[feature.get_normal_value_representation() for feature in diseases[0].features],
        index=[feature.title for feature in diseases[0].features],
        columns=["Нормальные значения (НЗ)"])

    df_normal_values.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=0, startcol=7, index=True)

    # TODO: сделать интексацию для заболеваний, пропихнув параметр извне
    df_clinical_picture = pd.DataFrame(data=[feature.title for feature in diseases[0].features * 2],
                                       index=[["Заболевание0"] * 6 + ["Заболевание1"] * 6],
                                       columns=["Клиническая картина (КК)"])

    df_clinical_picture.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=0, startcol=10, index=True)

    df_number_of_periods_of_dynamic = pd.DataFrame(data=[("Заболевание0",
                                                          feature.title,
                                                          feature.number_of_periods_of_dynamic)
                                                         for feature in diseases[0].features]).append(
        pd.DataFrame(
            data=[("Заболевание1",
                   feature.title,
                   feature.number_of_periods_of_dynamic)
                  for feature in diseases[1].features]
        )
    ) \
        .pivot_table(index=[0, 1], aggfunc="first")

    df_number_of_periods_of_dynamic.columns = ["Число периодов динамики (ЧПД)"]

    df_number_of_periods_of_dynamic.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=1,
                                             startcol=13, header=False)

    df_values_for_first_periods_of_dynamics = []
    for feature in diseases[0].features:
        for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
            df_values_for_first_periods_of_dynamics.append(
                (
                    "Заболевание0",
                    feature.title,
                    feature.number_of_periods_of_dynamic,
                    current_number_of_period_dynamic,
                    feature._get_values_for_periods_of_dynamic_representation(
                        index=current_number_of_period_dynamic)
                )
            )

    df_values_for_second_periods_of_dynamics = []
    for feature in diseases[1].features:
        for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
            df_values_for_second_periods_of_dynamics.append(
                (
                    "Заболевание1",
                    feature.title,
                    feature.number_of_periods_of_dynamic,
                    current_number_of_period_dynamic,
                    feature._get_values_for_periods_of_dynamic_representation(
                        index=current_number_of_period_dynamic)
                )
            )

    df_values_for_first_periods_of_dynamics = pd.DataFrame(
        data=df_values_for_first_periods_of_dynamics + df_values_for_second_periods_of_dynamics
    ).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc="first"
    )

    df_values_for_first_periods_of_dynamics.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=1,
                                                     startcol=17, header=False)

    df_upper_and_down_time_bound_for_first_disease = []
    for feature in diseases[0].features:
        for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
            df_upper_and_down_time_bound_for_first_disease.append(
                (
                    "Заболевание0",
                    feature.title,
                    feature.number_of_periods_of_dynamic,
                    current_number_of_period_dynamic,
                    feature.upper_and_down_time_bound[current_number_of_period_dynamic][0],
                    feature.upper_and_down_time_bound[current_number_of_period_dynamic][1]
                )
            )

    df_upper_and_down_time_bound_for_second_disease = []
    for feature in diseases[1].features:
        for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
            df_upper_and_down_time_bound_for_second_disease.append(
                (
                    "Заболевание1",
                    feature.title,
                    feature.number_of_periods_of_dynamic,
                    current_number_of_period_dynamic,
                    feature.upper_and_down_time_bound[current_number_of_period_dynamic][0],
                    feature.upper_and_down_time_bound[current_number_of_period_dynamic][1]
                )
            )

    df_upper_and_down_time_bound_for_first_disease += df_upper_and_down_time_bound_for_second_disease

    df_upper_and_down_time_bound = pd.DataFrame(data=df_upper_and_down_time_bound_for_first_disease).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc="first"
    )

    df_upper_and_down_time_bound.to_excel(writer, sheet_name="1. МБЗ", encoding="utf-8", startrow=1, startcol=23,
                                          header=False)


def make_second_report(first_diseases: [Disease], second_diseases: [Disease]):
    df_medicine_history_short_first = []
    for disease in first_diseases:
        df_medicine_history_without_concrete_values = []
        for feature in disease.features:
            for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
                df_medicine_history_without_concrete_values.append(
                    (
                        disease.medicine_history_title,
                        disease.title,
                        feature.title,
                        current_number_of_period_dynamic,
                        feature.duration_of_period_dynamic[current_number_of_period_dynamic],
                        feature.number_of_observation_moments[current_number_of_period_dynamic]
                    )
                )
        df_medicine_history_short_first.append(df_medicine_history_without_concrete_values)

    df_medicine_history_short_second = []
    for disease in second_diseases:
        df_medicine_history_without_concrete_values = []
        for feature in disease.features:
            for current_number_of_period_dynamic in range(feature.number_of_periods_of_dynamic):
                df_medicine_history_without_concrete_values.append(
                    (
                        disease.medicine_history_title,
                        disease.title,
                        feature.title,
                        current_number_of_period_dynamic,
                        feature.duration_of_period_dynamic[current_number_of_period_dynamic],
                        feature.number_of_observation_moments[current_number_of_period_dynamic]
                    )
                )
        df_medicine_history_short_second.append(df_medicine_history_without_concrete_values)

    df_medicine_history_short_first = sum(df_medicine_history_short_first, [])
    df_medicine_history_short_second = sum(df_medicine_history_short_second, [])

    df_medicine_history_short_first = pd.DataFrame(
        data=df_medicine_history_short_first).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc="first"
    )

    df_medicine_history_short_second = pd.DataFrame(
        data=df_medicine_history_short_second).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc="first"
    )

    df_medicine_history_short_first = df_medicine_history_short_first.append(df_medicine_history_short_second)

    df_medicine_history_short_first.to_excel(writer, sheet_name="2. МВД", encoding="utf-8", startrow=1,
                                             startcol=0, header=False)

    df_medicine_history_with_long_first = []
    for disease in first_diseases:
        df_medicine_history_with_concrete_values = []
        for feature in disease.features:
            tmp_concrete_time_values = sum(feature.concrete_moment_of_observation, [])
            for current_number_of_period_dynamic in range(len(tmp_concrete_time_values)):
                df_medicine_history_with_concrete_values.append(
                    (
                        disease.medicine_history_title,
                        disease.title,
                        feature.title,
                        tmp_concrete_time_values[current_number_of_period_dynamic],
                        feature.concrete_values_for_periods_of_dynamic_of_observation_moment[
                            current_number_of_period_dynamic]
                    )
                )
        df_medicine_history_with_long_first.append(df_medicine_history_with_concrete_values)

    df_medicine_history_with_long_second = []
    for disease in second_diseases:
        df_medicine_history_with_concrete_values = []
        for feature in disease.features:
            tmp_concrete_time_values = sum(feature.concrete_moment_of_observation, [])
            for current_number_of_period_dynamic in range(len(tmp_concrete_time_values)):
                df_medicine_history_with_concrete_values.append(
                    (
                        disease.medicine_history_title,
                        disease.title,
                        feature.title,
                        tmp_concrete_time_values[current_number_of_period_dynamic],
                        feature.concrete_values_for_periods_of_dynamic_of_observation_moment[
                            current_number_of_period_dynamic]
                    )
                )
        df_medicine_history_with_long_second.append(df_medicine_history_with_concrete_values)

    df_medicine_history_with_long_first = sum(df_medicine_history_with_long_first, [])
    df_medicine_history_with_long_second = sum(df_medicine_history_with_long_second, [])

    df_medicine_history_with_long_first = pd.DataFrame(data=df_medicine_history_with_long_first).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc="first"
    )

    df_medicine_history_with_long_second = pd.DataFrame(data=df_medicine_history_with_long_second).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc="first"
    )

    df_medicine_history_with_long_first = df_medicine_history_with_long_first.append(
        df_medicine_history_with_long_second)

    df_medicine_history_with_long_first.to_excel(writer, sheet_name="2. МВД", encoding="utf-8", startrow=1,
                                                 startcol=7, header=False)


def append_elements(elements):
    if len(elements) == 1:
        return str(elements.values[0])
    return " -> ".join(elements.values)


def make_third_report(medicine_history_first, good_alternatives_first, bad_alternatives_first,
                      medicine_history_second, good_alternatives_second, bad_alternatives_second):
    first_medicine_history_alternatives_report = []
    for history in medicine_history_first:
        for feature in history.features:
            if feature.type is not FeatureType.INTEGRAL:
                continue
            for feature_index, alternatives in enumerate(feature.possible_alternatives):
                if len(alternatives) == 1:
                    first_medicine_history_alternatives_report.append((
                        history.medicine_history_title,
                        history.title,
                        feature.title,
                        f"Альтернатива{len(alternatives)}.{feature_index}",
                        f"[1, {alternatives[-1]})"
                    ))
                for index in range(len(alternatives) - 1):
                    alternative_repr = str()
                    if index == 0:
                        alternative_repr = f"[1, {alternatives[index]})"
                        first_medicine_history_alternatives_report.append((
                            history.medicine_history_title,
                            history.title,
                            feature.title,
                            f"Альтернатива{len(alternatives)}.{feature_index}",
                            alternative_repr
                        ))
                    alternative_repr = f"[{alternatives[index]}, {alternatives[index + 1]})"
                    first_medicine_history_alternatives_report.append((
                        history.medicine_history_title,
                        history.title,
                        feature.title,
                        f"Альтернатива{len(alternatives)}.{feature_index}",
                        alternative_repr
                    ))
    first_medicine_history_alternatives_report = pd.DataFrame(
        data=first_medicine_history_alternatives_report).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc=append_elements
    )
    first_medicine_history_alternatives_report.to_excel(writer, sheet_name="3. ИФБЗ", encoding="utf-8", startrow=1,
                                                        startcol=0, header=False)

    second_medicine_history_alternatives_report = []
    for history in medicine_history_second:
        for feature in history.features:
            if feature.type is not FeatureType.INTEGRAL:
                continue
            for feature_index, alternatives in enumerate(feature.possible_alternatives):
                if len(alternatives) == 1:
                    second_medicine_history_alternatives_report.append((
                        history.medicine_history_title,
                        history.title,
                        feature.title,
                        f"Альтернатива{len(alternatives)}.{feature_index}",
                        f"[1, {alternatives[-1]})"
                    ))
                for index in range(len(alternatives) - 1):
                    alternative_repr = str()
                    if index == 0:
                        alternative_repr = f"[1, {alternatives[index]})"
                        second_medicine_history_alternatives_report.append((
                            history.medicine_history_title,
                            history.title,
                            feature.title,
                            f"Альтернатива{len(alternatives)}.{feature_index}",
                            alternative_repr
                        ))
                    alternative_repr = f"[{alternatives[index]}, {alternatives[index + 1]})"
                    second_medicine_history_alternatives_report.append((
                        history.medicine_history_title,
                        history.title,
                        feature.title,
                        f"Альтернатива{len(alternatives)}.{feature_index}",
                        alternative_repr
                    ))
    second_medicine_history_alternatives_report = pd.DataFrame(
        data=second_medicine_history_alternatives_report).pivot_table(
        index=[0, 1, 2, 3],
        aggfunc=append_elements
    )
    second_medicine_history_alternatives_report.to_excel(writer, sheet_name="3. ИФБЗ", encoding="utf-8", startrow=1,
                                                         startcol=6, header=False)

    index = 1
    for bad_table in bad_alternatives_first:
        bad_table = pd.DataFrame(data=bad_table)
        bad_table.to_excel(writer, sheet_name="3. ИФБЗ", encoding="utf-8", startrow=index, startcol=12, header=False)
        index += len(bad_table) + 1

    index = 1
    for good_table in good_alternatives_first:
        good_table = pd.DataFrame(data=good_table)
        good_table.to_excel(writer, sheet_name="3. ИФБЗ", encoding="utf-8", startrow=index, startcol=19,
                            header=False)
        index += len(good_table) + 1

    index = 1
    for bad_table in bad_alternatives_second:
        bad_table = pd.DataFrame(data=bad_table)
        bad_table.to_excel(writer, sheet_name="3. ИФБЗ", encoding="utf-8", startrow=index, startcol=26, header=False)
        index += len(bad_table) + 1

    index = 1
    for good_table in good_alternatives_second:
        good_table = pd.DataFrame(data=good_table)
        good_table.to_excel(writer, sheet_name="3. ИФБЗ", encoding="utf-8", startrow=index, startcol=33,
                            header=False)
        index += len(good_table) + 1


def make_alternatives_graphics(medicine_histories: [Disease]):
    image_names = []
    for feature_index in range(len(medicine_histories[0].features)):
        feature_table = []
        for medicine_history in medicine_histories:
            feature = medicine_history.features[feature_index]

            if feature.type is FeatureType.INTEGRAL:
                break

            x = sum(feature.concrete_moment_of_observation, [])
            y = feature.concrete_values_for_periods_of_dynamic_of_observation_moment
            for index, alternative in enumerate(feature.possible_alternatives):
                fig, ax = plt.subplots()
                ax.scatter(
                    x=x,
                    y=y
                )
                ax.grid(which="major")
                ax.set_xlabel("Моменты наблюдений(МН)")
                ax.set_ylabel("Значения в момент наблюдения(ЗМН)")
                title = f"{medicine_history.medicine_history_title}, {medicine_history.title}, {feature.title}, ЧПД{len(alternative)}, а{index:03d}"
                ax.set_title(title)
                for limit in alternative:
                    plt.axvline(limit, color="red")
                # plt.savefig(f"images/{title}")
                image_names.append(f"images/{title}.png")
    return image_names


class Alternative:
    def __init__(self,
                 feature,
                 medicine_history_title,
                 disease_title,
                 feature_title,
                 number_of_periods_of_dynamic,
                 alternative_index,
                 ):
        self.feature = feature
        self.medicine_history_title = medicine_history_title
        self.disease_title = disease_title
        self.feature_title = feature_title
        self.number_of_periods_of_dynamic = number_of_periods_of_dynamic
        self.alternative_index = alternative_index

    def __str__(self):
        return f"{self.medicine_history_title}, {self.disease_title}, {self.feature_title}, ЧПД{self.number_of_periods_of_dynamic}, а{self.alternative_index:03d}"

    def __repr__(self):
        return self.__str__()


def make_html_report(medicine_histories: [Disease]):
    html = """
        <html>
        <head>
        <title>IAD</title>
        </head>
        <body>
        </body>
        </html>
        """
    page = Soup(html, features="html.parser")
    body = page.find("body")
    body["name"] = "top"

    # Генерация навигации
    nav = page.new_tag("ul")
    for feature_index in range(len(medicine_histories[0].features)):
        feature = medicine_histories[0].features[feature_index]
        if feature.type is FeatureType.INTEGRAL:
            continue
        if medicine_histories[0].features[feature_index].type is FeatureType.BOOL:
            feature_type_title = f"Логический {medicine_histories[0].features[feature_index].title}"
        elif medicine_histories[0].features[feature_index].type is FeatureType.ENUM:
            feature_type_title = f"Перечислимый {medicine_histories[0].features[feature_index].title}"
        else:
            feature_type_title = f"Числовой {medicine_histories[0].features[feature_index].title}"
        feature_section_description = page.new_tag("li")
        feature_section_description.insert(0, feature_type_title)
        feature_section = page.new_tag("ul")
        nav.append(feature_section_description)
        nav.append(feature_section)
        for medicine_history in medicine_histories:
            feature = medicine_history.features[feature_index]
            medicine_history_section_description = page.new_tag("li")
            medicine_history_section_description.insert(0, medicine_history.medicine_history_title)
            medicine_history_section = page.new_tag("ul")
            feature_section.append(medicine_history_section_description)
            feature_section.append(medicine_history_section)
            for index, alternative in enumerate(feature.possible_alternatives):
                item = page.new_tag("li")
                reference = page.new_tag("a")
                reference_text = f"{medicine_history.medicine_history_title}, {medicine_history.title}, {feature.title}, ЧПД{len(alternative)}, а{index:03d}"
                reference.insert(0, reference_text)
                reference["href"] = "#" + reference_text
                item.append(reference)
                medicine_history_section.append(item)
    body.append(nav)

    # Генерация основного содержания страницы
    for feature_index in range(len(medicine_histories[0].features)):

        feature = medicine_histories[0].features[feature_index]
        if feature.type is FeatureType.INTEGRAL:
            continue

        paragraph = page.new_tag("p")
        paragraph["align"] = "center"
        paragraph["style"] = "font-size: 40px"

        feature_type_title = str()

        if medicine_histories[0].features[feature_index].type is FeatureType.BOOL:
            feature_type_title = f"Логический признак {medicine_histories[0].features[feature_index].title}"
        elif medicine_histories[0].features[feature_index].type is FeatureType.ENUM:
            feature_type_title = f"Перечислимые признак {medicine_histories[0].features[feature_index].title}"
        else:
            feature_type_title = f"Числовой признак {medicine_histories[0].features[feature_index].title}"

        paragraph.insert(0, feature_type_title)
        body.append(paragraph)
        body.append(page.new_tag("hr"))

        for medicine_history in medicine_histories:

            feature = medicine_history.features[feature_index]

            for index, alternative in enumerate(feature.possible_alternatives):
                image_path = f"{medicine_history.medicine_history_title}, {medicine_history.title}, {feature.title}, ЧПД{len(alternative)}, а{index:03d}"
                description = "Текстовое представление границ: " + feature.get_possible_alternative_representation(
                    index)

                paragraph = page.new_tag("p")
                paragraph["align"] = "center"
                paragraph["style"] = "font-size: 20px"
                paragraph["id"] = image_path
                image = page.new_tag("img")
                image["src"] = "images/" + image_path + ".png"

                paragraph.insert(0, image)
                paragraph.insert(1, page.new_tag("br"))
                paragraph.insert(1, page.new_tag("br"))
                paragraph.insert(3, description)
                paragraph.insert(4, page.new_tag("br"))
                paragraph.insert(4, page.new_tag("br"))
                paragraph.insert(4, page.new_tag("br"))

                anchor = page.new_tag("a")
                anchor.insert(0, "Наверх↑")
                anchor["href"] = "#top"

                paragraph.insert(7, anchor)
                body.append(paragraph)
                body.append(page.new_tag("hr"))

    with open("index.html", "w") as file:
        file.write(str(page))

    # Генерация массива альтернатив
    features_alternatives = list()
    for _ in range(len(medicine_histories[0].features)):
        features_alternatives.append([])
    for feature_index in range(len(medicine_histories[0].features)):
        feature = medicine_histories[0].features[feature_index]
        if feature.type is FeatureType.INTEGRAL:
            continue
        for medicine_history in medicine_histories:
            feature = medicine_history.features[feature_index]
            for index, alternative in enumerate(feature.possible_alternatives):
                features_alternatives[feature_index].append(
                    Alternative(
                        deepcopy(feature),
                        medicine_history.medicine_history_title,
                        medicine_history.title,
                        feature.title,
                        len(alternative),
                        index
                    )
                )
    return features_alternatives

good_table_mergee_count = 0
good_tables = []

def make_html_report_extended(alternatives: [Alternative]):
    html = """
            <html>
            <head>
            <title>IAD</title>
            </head>
            <body>
            </body>
            </html>
            """
    page = Soup(html, features="html.parser")
    body = page.find("body")
    body["name"] = "top"

    nav = page.new_tag("ul")
    body.append(nav)

    best_tables = []

    #for feature_index in range(len(alternatives)):
    for feature_index in tqdm(range(4)):
        section_description = page.new_tag("li")
        section_description.insert(0, f"Заболевание{feature_index}")
        nav.append(section_description)
        current_section_nav = page.new_tag("ul")
        nav.append(current_section_nav)
        current_splited_alternatives_by_medicine_history = []
        for _ in range(5):
            current_splited_alternatives_by_medicine_history.append([])
        for alternative in alternatives[feature_index]:
            current_splited_alternatives_by_medicine_history[int(alternative.medicine_history_title[-1])].append(
                alternative
            )
        for number_of_periods_of_dynamic in range(1, 6):
            current_splited_alternatives_by_number_of_periods_of_dynamic = []
            for _ in range(5):
                current_splited_alternatives_by_number_of_periods_of_dynamic.append([])
            for alternative_index, alternative in enumerate(current_splited_alternatives_by_medicine_history):
                for concrete_alternative in alternative:
                    if concrete_alternative.number_of_periods_of_dynamic == number_of_periods_of_dynamic:
                        current_splited_alternatives_by_number_of_periods_of_dynamic[alternative_index].append(
                            concrete_alternative
                        )
            for possible_alternatives_merge in itertools.product(*current_splited_alternatives_by_number_of_periods_of_dynamic):
                for max_merge_index in range(2, 6):
                    current_alternatives = possible_alternatives_merge[:max_merge_index]
                    #print("Соединяем альтернативы")

                    #Строим навигацию
                    merged_alternatives = page.new_tag("li")
                    a = page.new_tag("a")
                    a.insert(0, str(current_alternatives)[1:-1])
                    a["href"] = "#" + str(current_alternatives)[1:-1]
                    merged_alternatives.append(a)
                    current_section_nav.append(merged_alternatives)

                    #Добавляем основное содержание страницы
                    concat_description = page.new_tag("p")
                    concat_description["align"] = "center"
                    concat_description["style"] = "font-size: 20px"
                    concat_description["id"] = str(current_alternatives)[1:-1]
                    concat_description.insert(0, "Соединяем альтернативы: " + str(current_alternatives)[1:-1])
                    text_description = page.new_tag("pre")
                    text_description["align"] = "center"
                    text_description["style"] = "font-size: 20px"
                    alternative_image_container = page.new_tag("div")

                    body.append(concat_description)
                    concat_description.append(alternative_image_container)
                    concat_description.append(text_description)

                    merged_tables = []
                    for alternative in current_alternatives:
                        #print(alternative)

                        alternative_image = page.new_tag("img")
                        alternative_image["src"] = "images/" + str(alternative) + ".png"
                        alternative_image_container.append(alternative_image)

                        dynamic_periods = alternative.feature.possible_alternatives[alternative.alternative_index]
                        dynamic_periods_with_index = list(map(lambda x: sum(alternative.feature.concrete_moment_of_observation, []).index(x), dynamic_periods))
                        concrete_values_by_periods = alternative.feature.good_alternatives[tuple(dynamic_periods_with_index)]

                        table = []
                        for period_index in range(len(dynamic_periods)):
                            if period_index == 0:
                                table.append((period_index + 1, concrete_values_by_periods[period_index], dynamic_periods[period_index], dynamic_periods[period_index]))
                            else:
                                table.append((period_index + 1, concrete_values_by_periods[period_index],
                                              dynamic_periods[period_index] - dynamic_periods[period_index - 1], dynamic_periods[period_index] - dynamic_periods[period_index - 1]))
                        merged_tables.append(table)
                        #print(pd.DataFrame(data=table).to_markdown(index=False))
                        text_description.append(str(alternative) + "\n\n\n")
                        text_description.append(pd.DataFrame(
                            data=table,
                            columns=["Номер периода динамики", "Значения в период динамики", "НГ", "ВГ"]
                        ).to_markdown(index=False))
                        text_description.append("\n\n\n")
                    splited_rows_by_period_dynamic = []
                    for period_index in range(1, number_of_periods_of_dynamic + 1):
                        splited_rows_by_period_dynamic.append([])
                        for table in merged_tables:
                            for row in table:
                                if row[0] == period_index:
                                    splited_rows_by_period_dynamic[period_index - 1].append(row)
                    result_table = []
                    for period_index in range(1, number_of_periods_of_dynamic + 1):
                        union_of_values = set()
                        min_value_of_dynamic_period = 1_000_000
                        max_value_of_dynamic_period = -1_000_000
                        for current_row in splited_rows_by_period_dynamic[period_index - 1]:
                            union_of_values = union_of_values.union(current_row[1])
                            min_value_of_dynamic_period = min(min_value_of_dynamic_period, current_row[2])
                            max_value_of_dynamic_period = max(max_value_of_dynamic_period, current_row[2])
                        result_table.append((
                            period_index,
                            union_of_values,
                            min_value_of_dynamic_period,
                            max_value_of_dynamic_period
                        ))
                    is_bad_merge = False
                    for index in range(len(result_table) - 1):
                        if result_table[index][1].intersection(result_table[index + 1][1]) != 0:
                            is_bad_merge = True

                    if len(merged_tables) == 5 and not is_bad_merge:
                        best_tables.append((f"Заболевание{feature_index}", str(current_alternatives)[1:-1], result_table))

                    result_table = pd.DataFrame(
                        data=result_table,
                        columns=["Номер периода динамики", "Значения в период динамики", "НГ", "ВГ"]
                    ).to_markdown()
                    description_of_appending = page.new_tag("pre")
                    description_of_appending["style"] = "font-size: 20px;"
                    if is_bad_merge:
                        description_of_appending.insert(0, "Плохое сопоставление")
                        description_of_appending["style"] += "color: red;"
                    else:
                        description_of_appending.insert(0, "Хорошее сопоставление")
                        description_of_appending["style"] += "color: green;"
                    description_of_appending.append("\n\n\n" + result_table)
                    text_description.append(description_of_appending)
                    anchor = page.new_tag("a")
                    anchor.insert(0, "Наверх↑")
                    anchor["href"] = "#top"
                    text_description.append(anchor)
                    text_description.append(page.new_tag("hr"))
                    #print("RESULT:")
                    #print(result_table)
                    #print("-" * 30)
    best_tables_section = page.new_tag("li")
    best_tables_section.insert(0, "Лучшие альтернативы")
    nav.append(best_tables_section)
    best_table_nav = page.new_tag("ul")
    best_tables_section.append(best_table_nav)
    best_alternative_title = page.new_tag("pre")
    best_alternative_title["align"] = "center"
    best_alternative_title["style"] = "font-size: 30px;"
    best_alternative_title.insert(0, "Лучшие альтернативы")
    body.append(best_alternative_title)
    body.append(page.new_tag("hr"))
    for table in best_tables:
        row = page.new_tag("li")
        a = page.new_tag("a")
        a.insert(0, table[1])
        a["href"] = "#" + table[0] + table[1]
        row.append(a)
        best_table_nav.append(row)

    for table in best_tables:
        best_alternative_description = page.new_tag("pre")
        best_alternative_description["align"] = "center"
        best_alternative_description["style"] = "font-size: 20px; color: green;"
        best_alternative_description["id"] = table[0] + table[1]
        best_alternative_description.append(table[0] + "\n\n\n")
        best_alternative_description.append(table[1] + "\n\n\n")
        best_alternative_description.append(pd.DataFrame(
            data=table[2],
            columns=["Номер периода динамики", "Значения в период динамики", "НГ", "ВГ"]
        ).to_markdown() + "\n\n\n")
        anchor = page.new_tag("a")
        anchor.insert(0, "Наверх↑")
        anchor["href"] = "#top"
        best_alternative_description.append(anchor)
        best_alternative_description.append(page.new_tag("hr"))
        body.append(best_alternative_description)

    with open("alternative_concat.html", "w") as file:
        file.write(str(page))


def main():
    assert (pd is not None)
    assert (np is not None)

    first_disease = make_disease()
    second_disease = make_disease()
    second_disease.title = "Заболевание1"

    # make_first_report([first_disease, second_disease])

    medicine_history_array_first = [deepcopy(make_medicine_history(first_disease)) for _ in range(5)]
    for index, medicine_history in enumerate(medicine_history_array_first):
        medicine_history.set_medicine_history_title(f"ИБ{index}")
        generate_alternatives(medicine_history)

    medicine_history_array_second = [deepcopy(make_medicine_history(second_disease)) for _ in range(5)]
    for index, medicine_history in enumerate(medicine_history_array_second):
        medicine_history.set_medicine_history_title(f"ИБ{index}")
        generate_alternatives(medicine_history)

    # make_second_report(medicine_history_array_first, medicine_history_array_second)
    #
    # good_alternatives_first, bad_alternatives_first = reduce_alternatives_for_medicine_story(medicine_history_array_first)
    # good_alternatives_second, bad_alternatives_second = reduce_alternatives_for_medicine_story(medicine_history_array_second)
    #
    # make_third_report(medicine_history_array_first, good_alternatives_first, bad_alternatives_first,
    #                  medicine_history_array_second, good_alternatives_second, bad_alternatives_second)
    # image_names = make_alternatives_graphics(medicine_history_array_first)

    alternatives = make_html_report(medicine_history_array_first)
    make_html_report_extended(alternatives)

    # make_report_about_alternatives()
    # make_alternatives_graphics(medicine_history_array_second)


if __name__ == "__main__":
    # first_plot = plt.figure(1)
    # plt.scatter(x=[1, 2, 3, 4], y=['a', 'b', 'c', 'd'])
    # plt.grid(True)
    # plt.axvline(2.5, color="red")
    # second_plot = plt.figure(2)
    # plt.scatter(x=[1, 2, 3], y=['a', 'b', 'c'])
    # plt.grid(True)
    # plt.axvline(2.5, color="red")
    # plt.show()
    main()
    print(good_table_mergee_count)
    for element in good_tables:
        print(element[0])
        print(element[1])
    # writer.save()
    # print("ALTERNATIVES TOTAL:", global_alt)
# TODO: заоптимизировать код сопоставления альтернатив
# TODO: пофиксить багу с генерацией возможных значений и конкретных значений
