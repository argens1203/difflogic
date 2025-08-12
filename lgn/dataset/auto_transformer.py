from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod

from .converter import Converter


class AutoTransformer(ABC):
    # TODO: abstractmethod actually doesn't warn on missing implementation
    # @classproperty
    @classmethod
    @abstractmethod
    def attributes(cls) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def continuous_attributes(cls) -> set[str]:
        pass

    @classmethod
    @abstractmethod
    def discrete_attributes(cls) -> set[str]:
        pass

    @classmethod
    @abstractmethod
    def bin_sizes(cls) -> dict[str, int]:
        pass

    @classmethod
    def transform_label(cls, labels):
        if cls.label_encoder is None:
            cls.label_encoder = LabelEncoder()
            cls.label_encoder.fit(labels)

        return cls.label_encoder.transform(labels)

    @classmethod
    def transform_feature(cls, features):
        if cls.converter is None:
            cls.converter = Converter(
                attributes=cls.attributes(),
                continuous_attributes=cls.continuous_attributes(),
                discrete_attributes=cls.discrete_attributes(),
                bin_sizes=cls.bin_sizes(),
            )
            cls.converter.fit(features)

        return cls.converter.transform(features)

    @classmethod
    def inverse_transform_feature(cls, features):
        return cls.converter.inverse_transform(features)

    @classmethod
    def inverse_transform_label(cls, labels):
        return cls.label_encoder.inverse_transform(labels)

    # -- Getters -- #
    @classmethod
    def get_attribute_ranges(cls) -> list[int]:
        return cls.converter.get_attr_domains()

    @classmethod
    def get_input_dim(cls):
        return sum(cls.converter.get_attr_domains())

    @classmethod
    def get_num_of_classes(cls):
        return len(cls.label_encoder.classes_)
