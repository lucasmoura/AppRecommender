import sys

sys.path.insert(0, '..')

import apt
import commands
import nltk
import pandas as pd
import operator
import pickle
import re
import xapian

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.ml.data import MachineLearningData
from src.config import Config

USER_DATA_DIR = Config().user_data_dir


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
    pkgs_description = []
    pkgs_name = []
    ml_data = MachineLearningData()
    axi_path = "/var/lib/apt-xapian-index/index"
    axi = xapian.Database(axi_path)
    system_pkgs = get_system_pkgs()

    valid_pkgs = {}
    index = 3000

    with open(USER_DATA_DIR + 'pkgs_classifications.txt', 'ra') as data:
        pkg_data = pickle.load(data)

    with open('pc.txt', 'ra') as popcon:
        for result in popcon.readlines()[1:-2]:
            info = result.split(' ')
            pkg = info[2]

            if (pkg.startswith('lib') or pkg.endswith('data') or
                    pkg in system_pkgs or pkg.endswith('-data') or
                    pkg.endswith('doc')):
                continue

            valid_pkgs[pkg] = index
            index -= 1

    for pkg_name in pkg_data.keys():
        if pkg_name not in valid_pkgs:
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

    num_pkgs = len(pkgs_description)
    print 'Num packages: {0}'.format(num_pkgs)

    vectorizer = TfidfVectorizer(max_df=0.8,
                                 max_features=5000,
                                 min_df=3,
                                 stop_words='english',
                                 use_idf=True)

    pkg_features = vectorizer.fit_transform(pkgs_description)
    terms = vectorizer.get_feature_names()

    print pkg_features.shape
    num_clusters = 0
    max_value = -2

    for i in range(2, 60, 2):
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
        cluster_labels = kmeans.fit_predict(pkg_features)
        
        silhouette_avg = silhouette_score(pkg_features, cluster_labels)
        print("For n_clusters =", i,
            "The average silhouette_score is :", silhouette_avg)

        if silhouette_avg > max_value:
            max_value = silhouette_avg
            num_clusters = i

    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(pkg_features)
    clusters = kmeans.labels_.tolist()

    user_pkgs = {'pkg': pkgs_name, 'description': pkgs_description,
                 'clusters': clusters}

    frame = pd.DataFrame(user_pkgs, index=[clusters],
                         columns=['pkg', 'description', 'clusters'])

    print frame['clusters'].value_counts()

    preferred_clusters = {}
    for i in range(num_clusters):
        preferred_clusters[i] = 0

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
        cluster_pkgs = frame.ix[i]['pkg']
        
        if type(cluster_pkgs) == str:
            cluster_pkgs = [cluster_pkgs]
        else:
            cluster_pkgs = cluster_pkgs.values.tolist()
        
        num_pkgs = len(cluster_pkgs)
        for pkg in cluster_pkgs:
            print(' %s,' % pkg)
            preferred_clusters[i] += valid_pkgs[pkg]

        preferred_clusters[i] /= num_pkgs
        print '\n'
        print '\n'

    sorted_clusters = sorted(preferred_clusters.items(),
                             key=operator.itemgetter(1))
    index = [i for i,j in sorted_clusters]
    for i in reversed(index):
        print 'Cluster {}: {}'.format(i, preferred_clusters[i])


if __name__ == '__main__':
    main()
