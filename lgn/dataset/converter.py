from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder


class Converter:
    def __init__(
        self,
        attributes,
        continuous_attributes=None,
        discrete_attributes=None,
        bin_sizes=None,
    ):
        self.attributes = attributes
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.bin_sizes = bin_sizes

        if continuous_attributes is None:
            self.continuous_attributes = set(attributes) - discrete_attributes
        if discrete_attributes is None:
            self.discrete_attributes = set(attributes) - continuous_attributes

        self.convertors = dict()
        self.setup_convertors()

        self.ohe = None  # Note: OneHotEncoder is not initialized until fit is called
        self.n_classes = []

    def setup_convertors(self):
        for attr in self.attributes:
            if attr in self.continuous_attributes:
                self.convertors[attr] = KBinsDiscretizer(
                    n_bins=self.bin_sizes[attr], encode="ordinal", strategy="kmeans"
                )
            elif attr in self.discrete_attributes:
                self.convertors[attr] = LabelEncoder()
            else:
                raise ValueError(f"Unknown attribute {attr}")

    def transform_attr(self, data, attr):
        if attr in self.continuous_attributes:
            return (
                self.convertors[attr]
                .transform(data.astype(float).reshape(-1, 1))
                .reshape(-1)
            )
        elif attr in self.discrete_attributes:
            return self.convertors[attr].transform(data)
        else:
            raise ValueError(f"Unknown attribute {attr}")

    def fit_attr(self, data, attr):
        converter = self.convertors[attr]

        if attr in self.continuous_attributes:
            data = data.astype(float).reshape(-1, 1)
            converter.fit(data)
            return converter.n_bins_[0].item()

        if attr in self.discrete_attributes:
            converter.fit(data)
            return len(converter.classes_)

        raise ValueError(f"Unknown attribute {attr}")

    def inverse_transform_attr(self, data, attr):
        converter = self.convertors[attr]

        if attr in self.continuous_attributes:
            data = data.reshape(-1, 1)
            return converter.inverse_transform(data).reshape(-1)

        if attr in self.discrete_attributes:
            return converter.inverse_transform(data)

        raise ValueError(f"Unknown attribute {attr}")

    def transform(self, data):
        for i, attr in enumerate(self.attributes):
            data[:, i] = self.transform_attr(data[:, i], attr)

        data = data.astype(float).astype(int)

        if self.ohe is None:
            self.ohe = OneHotEncoder(
                sparse_output=False,
            )
            data = self.ohe.fit_transform(data)
        else:
            data = self.ohe.transform(data)

        return data

    def inverse_transform(self, data):
        data = self.ohe.inverse_transform(data)
        data = data.astype(str)
        for i, attr in enumerate(self.attributes):
            data[:, i] = self.inverse_transform_attr(data[:, i].astype(int), attr)
        return data

    def fit(self, data):
        n_classes = []
        for i, attr in enumerate(self.attributes):
            n = self.fit_attr(data[:, i], attr)
            n_classes.append(n)

        self.n_classes = n_classes

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
