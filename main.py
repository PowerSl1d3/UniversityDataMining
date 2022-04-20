import numpy as np
import pandas as pd
from enum import Enum
import random
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm

random.seed(13)

writer = pd.ExcelWriter("ИАД Аксененко Олег.xls", engine="openpyxl")

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
        FeatureType.INTEGRAL,
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

    df_medicine_history_with_long_first = df_medicine_history_with_long_first.append(df_medicine_history_with_long_second)

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
    first_medicine_history_alternatives_report = pd.DataFrame(data=first_medicine_history_alternatives_report).pivot_table(
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


def main():
    assert (pd is not None)
    assert (np is not None)

    first_disease = make_disease()
    second_disease = make_disease()
    second_disease.title = "Заболевание1"

    make_first_report([first_disease, second_disease])

    medicine_history_array_first = [deepcopy(make_medicine_history(first_disease)) for _ in range(5)]
    for index, medicine_history in enumerate(medicine_history_array_first):
        medicine_history.set_medicine_history_title(f"ИсторияБолезни{index}")
        generate_alternatives(medicine_history)

    medicine_history_array_second = [deepcopy(make_medicine_history(second_disease)) for _ in range(5)]
    for index, medicine_history in enumerate(medicine_history_array_second):
        medicine_history.set_medicine_history_title(f"ИсторияБолезни{index}")
        generate_alternatives(medicine_history)

    make_second_report(medicine_history_array_first, medicine_history_array_second)

    good_alternatives_first, bad_alternatives_first = reduce_alternatives_for_medicine_story(medicine_history_array_first)
    good_alternatives_second, bad_alternatives_second = reduce_alternatives_for_medicine_story(medicine_history_array_second)

    make_third_report(medicine_history_array_first, good_alternatives_first, bad_alternatives_first,
                      medicine_history_array_second, good_alternatives_second, bad_alternatives_second)


if __name__ == "__main__":
    main()
    writer.save()
    print("ALTERNATIVES TOTAL:", global_alt)
