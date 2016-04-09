import sys

sys.path.insert(0, '..')

import apt
import commands
import nltk
import pandas as pd
import re
import xapian

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from src.ml.pkg_time import get_packages_from_apt_mark
from src.ml.data import MachineLearningData


def stem_description(pkg_description):
    stemmer = SnowballStemmer("english")
    tokens = [word for sent in nltk.sent_tokenize(pkg_description) for word in
              nltk.word_tokenize(sent)]
    filtered_tokens = []

    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def filter_description(pkg_description, stop_words):
    description = re.sub('[^a-zA-Z]', ' ', pkg_description)
    description = description.lower()
    description = description.split()

    description = [word for word in description if word not in stop_words]
    return description


def filter_debtags(ml_data, axi, pkg_name):
    debtags = ml_data.get_pkg_debtags(axi, pkg_name)
    debtags = ml_data.filter_debtags(debtags)
    return debtags


def get_system_pkgs():
    system_pkgs = set()
    all_pkgs = commands.getoutput("dpkg-query -Wf \
                                    '${Package;-40}${Priority}\n'")

    priority_terms = set(['important', 'required', 'standard'])
    languages_terms = set(['python', 'perl'])

    for line in all_pkgs.splitlines():
        line_split = line.split(' ')
        pkg_name = line_split[0]
        pkg_priority = line_split[-1].strip()

        if (pkg_priority in priority_terms and
                pkg_name not in languages_terms):
            system_pkgs.add(pkg_name)

    return system_pkgs


def main():

    cache = apt.Cache()
    manual_pkgs = get_packages_from_apt_mark()
    pkgs_description = []
    pkgs_name = []
    ml_data = MachineLearningData()
    axi_path = "/var/lib/apt-xapian-index/index"
    axi = xapian.Database(axi_path)
    system_pkgs = get_system_pkgs()

    for index, pkg_name in enumerate(manual_pkgs):
        if (pkg_name.startswith('lib') or pkg_name.endswith('doc') or
                pkg_name in system_pkgs):
            continue

        pkg = cache[pkg_name].versions[0]

        description = pkg.description.strip()
        description = stem_description(description)

        section = pkg.section

        debtags = filter_debtags(ml_data, axi, pkg_name)

        description.append(section)

        for debtag in debtags:
            description.append(debtag)

        pkgs_description.append(' '.join(description))
        pkgs_name.append(pkg_name)

    print 'Num packages: {0}'.format(len(pkgs_description))

    vectorizer = TfidfVectorizer(max_df=0.8,
                                 max_features=5000,
                                 min_df=3,
                                 stop_words='english',
                                 use_idf=True,
                                 ngram_range=(1, 3))

    pkg_features = vectorizer.fit_transform(pkgs_description)
    terms = vectorizer.get_feature_names()

    print pkg_features.shape

    num_clusters = 8
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(pkg_features)

    clusters = kmeans.labels_.tolist()

    user_pkgs = {'pkg': pkgs_name, 'description': pkgs_description,
                 'clusters': clusters}

    frame = pd.DataFrame(user_pkgs, index=[clusters],
                         columns=['pkg', 'description', 'clusters'])

    print frame['clusters'].value_counts()

    print("Top terms per cluster:")
    print()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        print("Cluster {} words:".format(i))
        for ind in order_centroids[i, :6]:
            print(' %s' % terms[ind].encode('utf-8', 'ignore'))
        print '\n'
        print '\n'
        print("Cluster %d pkgs:" % i)
        for pkg in frame.ix[i]['pkg'].values.tolist():
            print(' %s,' % pkg)
        print '\n'
        print '\n'


if __name__ == '__main__':
    main()
