"""Global Vector Embeddings.
"""
"""
Copyright 2017 Parag K. Mital.  See also NOTICE.md.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import matplotlib.pyplot as plt
from cadl import utils
import zipfile
from scipy.spatial import distance, distance_matrix
from sklearn.decomposition import PCA


def get_model():
    """Summary

    Returns
    -------
    TYPE
        Description
    """
    # Download the glove model and open a zip file
    file = utils.download('http://nlp.stanford.edu/data/wordvecs/glove.6B.zip')
    zf = zipfile.ZipFile(file)

    # Collect the words and their vectors
    words = []
    vectors = []
    for l in zf.open("glove.6B.300d.txt"):
        t = l.strip().split()
        words.append(t[0].decode())
        vectors.append(list(map(np.double, t[1:])))

    # Store as a lookup table
    wordvecs = np.asarray(vectors, dtype=np.double)
    word2id = {word: i for i, word in enumerate(words)}
    return wordvecs, word2id, words


def course_example():
    """Summary
    """
    wordvecs, word2id, words = get_model()

    word = '2000'
    print(word2id[word])

    print(wordvecs[word2id[word]])

    # Get distances to target word
    target_vec = wordvecs[word2id[word]]
    dists = []
    for vec_i in wordvecs:
        dists.append(distance.cosine(target_vec, vec_i))

    k = 20

    # Print top nearest words
    idxs = np.argsort(dists)
    for idx_i in idxs[:k]:
        print(words[idx_i], dists[idx_i])

    # Plot top nearest words
    labels = [words[idx_i] for idx_i in idxs[:k]]
    plt.figure()
    plt.bar(range(k),
            [dists[idx_i] for idx_i in idxs[:k]])
    ax = plt.gca()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical')
    plt.xlabel('label')
    plt.ylabel('distances')

    # Create distance matrix
    vecs = [wordvecs[idx_i] for idx_i in idxs[:k]]
    dm = distance_matrix(vecs, vecs)
    plt.figure()
    plt.imshow(dm)
    ax = plt.gca()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_yticklabels(labels)
    plt.colorbar()

    # Plot data points in reduced dimensionality using principal components
    # of the distance matrix
    res = PCA(2).fit_transform(dm / np.mean(dm, axis=0, keepdims=True))
    pc1, pc2 = res[:, 0], res[:, 1]
    plt.figure()
    plt.scatter(pc1, pc2)
    for i in range(len(labels)):
        plt.text(pc1[i], pc2[i], labels[i])

    # Let's stick it all in a function and explore some other words:
    def plot_nearest_words(word, k=20):
        """Summary

        Parameters
        ----------
        word : TYPE
            Description
        k : int, optional
            Description
        """
        # Get distances to target word
        target_vec = wordvecs[word2id[word]]
        dists = []
        for vec_i in wordvecs:
            dists.append(distance.cosine(target_vec, vec_i))
        idxs = np.argsort(dists)
        labels = [words[idx_i] for idx_i in idxs[:k]]
        vecs = [wordvecs[idx_i] for idx_i in idxs[:k]]
        dm = distance_matrix(vecs, vecs)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Create distance matrix
        axs[0].imshow(dm)
        axs[0].set_xticks(range(len(labels)))
        axs[0].set_yticks(range(len(labels)))
        axs[0].set_xticklabels(labels, rotation='vertical')
        axs[0].set_yticklabels(labels)

        # Center the distance matrix
        dm = dm / np.mean(dm, axis=0, keepdims=True)

        # Plot data points in reduced dimensionality using principal components
        # of the distance matrix
        res = PCA(2).fit_transform(dm)
        pc1, pc2 = res[:, 0], res[:, 1]
        axs[1].scatter(pc1, pc2)
        for i in range(len(labels)):
            axs[1].text(pc1[i], pc2[i], labels[i])

    plot_nearest_words('2000')
    plot_nearest_words('intelligence')

    # What else can we explore?  Well this embedding is "linear" meaning we can
    # actually try performing arithmetic in this space.  A classic example is what
    # happens when we perform: "man" - "king" + "woman"?  Or in other words, can the
    # word embedding understand analogies?  For instance, if man is to king as woman
    # is to queen, then we should be able to subtract man and king, and add woman
    # to see the result of the analogy.

    # Let's create a function which will return us the nearest words rather than
    # plot them:
    def get_nearest_words(target_vec, k=20):
        """Summary

        Parameters
        ----------
        target_vec : TYPE
            Description
        k : int, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        # Get distances to target vector
        dists = []
        for vec_i in wordvecs:
            dists.append(distance.cosine(target_vec, vec_i))
        # Get top nearest words
        idxs = np.argsort(dists)
        res = []
        for idx_i in idxs[:k]:
            res.append((words[idx_i], dists[idx_i]))
        return res

    # And a convenience function for returning a vector
    def get_vector(word):
        """Summary

        Parameters
        ----------
        word : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        return wordvecs[word2id[word]]

    # Now we can try some word embedding arithmetic
    get_nearest_words(get_vector('king') - get_vector('man') + get_vector('woman'))
    get_nearest_words(get_vector('france') - get_vector('french') + get_vector('spain'))
