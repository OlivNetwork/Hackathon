# Label encoding:
# Data transformation technique where each category is assigned a unique number instead of
# a qualitative value.

from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder()
encoder = LabelEncoder()

data = [1, 2, 2, 6]

# Fit to the data
encoder.fit(data)

# Transform the data
transformed = encoder.transform(data)

# Reverse the transformation
inverse = encoder.inverse_transform(transformed)


# Some potential problems with label encoding:
# Imagine you’re analyzing a dataset with categories of music genres. You label encode “Blues,”
# “Electronic Dance Music (EDM),” “Hip Hop,” “Jazz,” “K-Pop,” “Metal,” “ and “Rock,” with the
# following numeric values, “1, 2, 3, 4, 5, 6, and 7.”
#
# With this label encoding, the resulting machine learning model could derive not only a ranking,
# but also a closer connection between Blues (1) and EDM (2) because of how close they are numerically
# than, say, Blues(1) and Jazz(4). In addition to these presumed relationships (which you may or may
# not want in your analysis) you should also notice that each code is equidistant from the other in the
# numeric sequence, as in 1 to 2 is the same distance as 5 to 6, etc. The question is, does that
# equidistant relationship accurately represent the relationships between the music genres in your
# dataset? To ask another question, after encoding, will the visualization or model you build treat the
# encoded labels as a ranking?
#
# The same could be said for the mushroom example above. After label encoding mushroom types, are you
# satisfied with the fact that the mushrooms are now in a presumed ranked order with button mushrooms
# ranked first and toadstool ranked eighth?
#
# In summary, label encoding may introduce unintended relationships between the categorical data in your
# dataset. When you are making decisions about label encoding, consider the algorithm you’ll apply to the
# data and how it may or may not impact label encoded categorical data.
#
# Fortunately, there is another method for categorical encoding that may help with these potential problems.
#

# One-hot encoding:
# The idea is to create a new column for each category type, then for each value indicate a 0 or a 1 — 0
# meaning, no, and 1 meaning, yes.
#
# Dummy variables
# Variables with values of 0 or 1, which indicate the presence or absence of something.

pd.get_dummies()

# This creation of dummies is called one-hot encoding.

#
# Use one-hot encoding when:
#
# A. There is a relatively small amount of categorical variables — because one-hot encoding uses much more
# data than label encoding.
# B. The categorical variables have no particular order
# C. You use a machine learning model in combination with dimensionality reduction (like Principal Component
# Analysis (PCA))
#

# MORE DISCUSSION:
# We may suggest the project to use WoE encoder/ Ordinal encoder instead of label encoding in sklearn. The
# [category encoder]() library is a better pipeline compatible package for such purposes. This package hosts
# several encoders like Helmart encoder, Leave one out encoder, WOE encoder, etc. that could be used directly
# in a pipeline without much modification

# We usually encode a category column based on the domain and the number of categories. If a column has too many
# categories, I suggest to bin them to 4-5 categories first and then encode them using one-hot representation. We
# sometimes rely on frequency counts/ domain experience too as an alternative to a label encoding (ready-made
# solution). This accounts for data and assignment specific encoding rather than the usage of a standard method
# that may elicit relations not in tandem with the end-user's expectations.

















