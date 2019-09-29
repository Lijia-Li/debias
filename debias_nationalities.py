import os
from pathlib import Path
from itertools import islice

import numpy as np
from sklearn.decomposition import PCA

from experiments import read_word_list, read_gender_pairs
# from experiments import measure_projection_bias, measure_analogy_bias
from experiments import WordEmbedding
from experiments import debias_bolukbasi, _bolukbasi_debias, _bolukbasi_equalize, redirect_stderr, normalize, recenter


def define_pca_target_direction(embedding, gender_pairs):
    """Calculate the target direction using PCA.

    Parameters:
        embedding (WordEmbedding): A word embedding.
        gender_pairs (Iterable[Tuple[str, str]]): A list of male-female word pairs.

    Returns:
        numpy.ndarray: A female->male vector.

    Raises:
        ValueError: If none of the target pairs are in the embedding.
    """
    matrix = []
    for male_word, female_word in gender_pairs:
        if male_word not in embedding or female_word not in embedding:
            continue
        matrix.extend(recenter(
            np.array([embedding[male_word], embedding[female_word]])
        ))
    if not matrix:
        raise ValueError('embedding does not contain any gender pairs.')
    matrix = np.array(matrix)
    pca = PCA(n_components=10)
    pca.fit(matrix)
    gender_direction = normalize(pca.components_[0])
    return align_gender_direction(embedding, gender_direction, gender_pairs)


def align_gender_direction(embedding, gender_direction, gender_pairs):
    """Make sure the direction is female->male, not vice versa.

    Parameters:

    Parameters:
        embedding (WordEmbedding): A word embedding.
        gender_direction (numpy.ndarray): A male->female or female->male vector.
        gender_pairs (Iterable[Tuple[str, str]]): A list of male-female word pairs.

    Returns:
        numpy.ndarray: A female->male vector.
    """
    total = 0
    for male_word, female_word in gender_pairs:
        if male_word not in embedding or female_word not in embedding:
            continue
        male_vector = embedding[male_word]
        female_vector = embedding[female_word]
        total += (male_vector - female_vector).dot(gender_direction)
    if total < 0:
        gender_direction = -gender_direction
    return gender_direction


def cumulative_sum(seq):
    """Calculate the cumulative sum of a sequence of numbers.

    Parameters:
        seq (Sequence[float]): The sequence of numbers.

    Yields:
        float: The cumulative sum.
    """
    total = 0
    for x in seq:
        total += x
        yield total


def find_thresholds(thresholds, sequence):
    threshold_index = 0
    var_index = 0
    while threshold_index < len(thresholds) and var_index < len(sequence):
        variance = sequence[var_index]
        if variance > thresholds[threshold_index]:
            yield var_index
            threshold_index += 1
        var_index += 1


# todo: refactor/formatting the function
def define_nationality_dimensions(embedding, nationalities, thresholds):
    print("in the function")
    # extract embeddings for each word in the set
    matrix = [
        list(embedding[nationality.lower()])
        for nationality in nationalities
        if nationality.lower() in embedding
    ]
    # recenter the saved embeddings --> equal the distance between center and the embedding
    centered = recenter(np.array(matrix))
    # do PCA analysis
    pca = PCA(n_components=min(len(matrix), len(matrix[0])))
    pca.fit(centered)
    # compile a explained variance list
    variance_percent_ls = [variance_percent for component, variance_percent in islice(zip(pca.components_, pca.explained_variance_ratio_), 20)]
    # print("variance percents:", variance_percent_ls)
    # check cumulative explained variance
    cum_var = list(cumulative_sum(pca.explained_variance_ratio_))
    # print("cumulative variable:",cum_var)

    # find the position to which the combined variance can explain to the level of threshold
    index_explained_var = list(find_thresholds(thresholds, cum_var))
    print("threshold:", index_explained_var)
    # compile subspace for each threshold
    for idx, position in enumerate(index_explained_var):
        # transpose the components to match the embedding vector sieze
        components = pca.components_[0:position]
        print(components.shape)
        threshold = thresholds[idx]
        yield threshold, components


def main():
    # load original model
    standard_model = WordEmbedding.load_fasttext_file(Path('models/wikipedia-1.fasttext.cbow.bin'))
    print("loaded standard_model")
    # load nationalities words set
    nationalities_set = read_word_list(Path('data/swap-groups/nationalities'))

    # make up a few thresholds (20% 40% 60% 80%)
    thresholds = [n / 10 for n in range(2, 10, 2)]
    # find subspace
    subspaces = define_nationality_dimensions(standard_model, nationalities_set, thresholds)
    for threshold, subspace in subspaces:
        # initiate output path
        print("start creating model, threshold being", threshold, "reducing", subspace.shape[0], "dimentsions")
        out_path = Path('models/nationality--debias-' + str(threshold) + '.w2v')
        # create the model by reducing nationality subspace(s)
        debias_bolukbasi(standard_model, subspace, out_path)


if __name__ == "__main__":
    main()

# todo run the model
# todo the analysis created model
