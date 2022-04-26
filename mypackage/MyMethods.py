import warnings
warnings.filterwarnings('ignore')
from pandas.core.common import flatten
from sklearn.cluster import DBSCAN
from ots_eval.stability_evaluation.close import CLOSE
import statistics
import pandas as pd
import numpy as np

#Allgemeine Methoden

def get_missing_rows(dataframe, df_id_name, df_time_name):
    id_list = list(set(getattr(dataframe, df_id_name)))
    time_list = list(set(getattr(dataframe, df_time_name)))
    result = []

    for id_elem in id_list:
        for time_elem in time_list:
            if dataframe[(dataframe[df_id_name] == id_elem) & (dataframe[df_time_name] == time_elem)].empty:
                result.append([id_elem, time_elem])
    
    return result  


def id_clusters_list(dataframe, missing_id_list, df_id_name, df_cluster_name):
    id_cluster_list = []
    needed_clusters = []

    for id_elem in missing_id_list:
        needed_clusters = list(set(getattr(dataframe[dataframe[df_id_name] == id_elem], df_cluster_name)))
        if -1 in needed_clusters:
            needed_clusters.remove(-1)
        id_cluster_list.append([id_elem, list(needed_clusters)])
        needed_clusters.clear()

    return id_cluster_list


#Most_Frequent_Cluster_Member

def most_frequent_cluster_member(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    id_list = list(set(getattr(dataframe, df_id_name)))
    id_cluster_list = id_clusters_list(dataframe, id_list, df_id_name, df_cluster_name)
    mfcm = id_mfcm_list(dataframe, missing_rows, id_list, id_cluster_list, df_id_name, df_time_name)
    middle = []
    result = []
    
    for missing_id_elem, missing_time_elem, assigned_mfcm in mfcm:
        for member_id in assigned_mfcm:
            middle.append(getattr(dataframe[(dataframe[df_id_name] == member_id) & (dataframe[df_time_name] == missing_time_elem)], df_feature_name).values)
        result.append([missing_id_elem, missing_time_elem, statistics.median(middle)[0]])
        middle.clear()
    
    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "mfcm"])

def id_mfcm_list(dataframe, missing_rows, id_list, id_cluster_list, df_id_name, df_time_name):
    most_frequent_cluster_member = []
    mfcm_ids = []
    current_count = 0
    biggest_count = 0

    for missing_id_elem, missing_time_elem in missing_rows:
        for id_elem in getattr(dataframe[dataframe[df_time_name] == missing_time_elem], df_id_name).values:
            if missing_id_elem != id_elem:
                current_count = len(set(id_cluster_list[id_list.index(missing_id_elem)][1]).intersection(id_cluster_list[id_list.index(id_elem)][1]))
                if current_count == biggest_count:
                    mfcm_ids.append(id_elem)
                if current_count > biggest_count:
                    biggest_count = current_count
                    mfcm_ids.clear()
                    mfcm_ids.append(id_elem)
        most_frequent_cluster_member.append([missing_id_elem, missing_time_elem, list(mfcm_ids)])
        mfcm_ids.clear()
        biggest_count = 0
    
    return most_frequent_cluster_member

    
#nearest_Most_Frequent_Cluster_Member

def most_frequent_cluster_member_nearest(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    id_list = list(set(getattr(dataframe,df_id_name)))
    id_cluster_list = id_clusters_list(dataframe, id_list, df_id_name, df_cluster_name)
    mfcm = id_mfcm_list_nearest(dataframe, missing_rows, id_list, id_cluster_list, df_id_name, df_time_name)
    current_dist = 0
    smallest_dist = float('inf')
    middle = 0
    result = []
    
    for missing_id_elem, missing_time_elem, assigned_mfcm in mfcm:
        for member_id, clusters in assigned_mfcm:
            current_dist = calc_distances(dataframe, missing_id_elem, member_id, clusters, df_id_name, df_feature_name, df_cluster_name)
            if current_dist < smallest_dist:
                smallest_dist = current_dist
                middle = getattr(dataframe[(dataframe[df_id_name] == member_id) & (dataframe[df_time_name] == missing_time_elem)], df_feature_name).values
        result.append([missing_id_elem, missing_time_elem, middle[0]])
        smallest_dist = float('inf')
    
    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "mfcm_nearest"])


def id_mfcm_list_nearest(dataframe, missing_rows, id_list, id_cluster_list, df_id_name, df_time_name):
    most_frequent_cluster_member = []
    mfcm_ids = []
    current_count = 0
    biggest_count = 0

    for missing_id_elem, missing_time_elem in missing_rows:
        for id_elem in getattr(dataframe[dataframe[df_time_name] == missing_time_elem], df_id_name).values:
            if missing_id_elem != id_elem:
                current_count = len(set(id_cluster_list[id_list.index(missing_id_elem)][1]).intersection(id_cluster_list[id_list.index(id_elem)][1]))
                if current_count == biggest_count:
                    mfcm_ids.append([id_elem, list(set(id_cluster_list[id_list.index(missing_id_elem)][1]).intersection(id_cluster_list[id_list.index(id_elem)][1]))])
                if current_count > biggest_count:
                    biggest_count = current_count
                    mfcm_ids.clear()
                    mfcm_ids.append([id_elem, list(set(id_cluster_list[id_list.index(missing_id_elem)][1]).intersection(id_cluster_list[id_list.index(id_elem)][1]))])
        most_frequent_cluster_member.append([missing_id_elem, missing_time_elem, list(mfcm_ids)])
        mfcm_ids.clear()
        biggest_count = 0
    
    return most_frequent_cluster_member

def calc_distances(dataframe, missing_id, member_id, clusters, df_id_name, df_feature_name, df_cluster_name):
    result = []
    
    for cluster in clusters:
        x = getattr(dataframe[(dataframe[df_id_name] == missing_id) & (dataframe[df_cluster_name] == cluster)], df_feature_name).values
        y = getattr(dataframe[(dataframe[df_id_name] == member_id) & (dataframe[df_cluster_name] == cluster)], df_feature_name).values
        result.append(abs(x-y))

    return sum(result)

#New-Method-Mean

def cluster_id_mean_list(dataframe, df_feature_name, df_cluster_name):
    cluster_list = list(set(getattr(dataframe, df_cluster_name)))
    result = []
    
    for cluster in cluster_list:
        result.append([cluster, statistics.mean(getattr(dataframe[dataframe[df_cluster_name] == cluster], df_feature_name))])
    
    return result

def id_mean_of_clusters(dataframe, df_id_name, df_feature_name, df_cluster_name):
    id_list = list(set(getattr(dataframe, df_id_name)))
    id_cluster_list = id_clusters_list(dataframe, id_list, df_id_name, df_cluster_name)
    cluster_mean = cluster_id_mean_list(dataframe, df_feature_name, df_cluster_name)
    indexing_list = [item[0] for item in cluster_mean]
    count = []
    result = []

    for id_elem, clusters in id_cluster_list:
        for cluster in clusters:
            count.append(cluster_mean[indexing_list.index(cluster)][1])
        result.append([id_elem, statistics.mean(count)])
        count.clear()

    return result


def new_method_mean(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    id_mean = id_mean_of_clusters(dataframe, df_id_name, df_feature_name, df_cluster_name)
    indexing = [item[0] for item in id_mean]
    result = []
    
    for missing_id_elem, missing_time_elem in missing_rows:
        result.append([missing_id_elem, missing_time_elem, id_mean[indexing.index(missing_id_elem)][1]])
    
    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "new_method_mean"])

#New-Method-Median

def cluster_id_median_list(dataframe, df_feature_name, df_cluster_name):
    cluster_list = list(set(getattr(dataframe, df_cluster_name)))
    result = []
    
    for cluster in cluster_list:
        result.append([cluster, statistics.median(getattr(dataframe[dataframe[df_cluster_name] == cluster], df_feature_name))])
    
    return result

def id_median_of_clusters(dataframe, df_id_name, df_feature_name, df_cluster_name):
    id_list = list(set(getattr(dataframe, df_id_name)))
    id_cluster_list = id_clusters_list(dataframe, id_list, df_id_name, df_cluster_name)
    cluster_median = cluster_id_median_list(dataframe, df_feature_name, df_cluster_name)
    indexing_list = [item[0] for item in cluster_median]
    count = []
    result = []

    for id_elem, clusters in id_cluster_list:
        for cluster in clusters:
            count.append(cluster_median[indexing_list.index(cluster)][1])
        result.append([id_elem, statistics.median(count)])
        count.clear()

    return result

def new_method_median(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    id_median = id_median_of_clusters(dataframe, df_id_name, df_feature_name, df_cluster_name)
    indexing = [item[0] for item in id_median]
    result = []
    
    for missing_id_elem, missing_time_elem in missing_rows:
        result.append([missing_id_elem, missing_time_elem, id_median[indexing.index(missing_id_elem)][1]])
    
    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "new_method_median"])

#New-Method-Mode

def cluster_id_mode_list(dataframe, df_feature_name, df_cluster_name):
    cluster_list = list(set(dataframe[df_cluster_name]))
    result = []
    
    for cluster in cluster_list:
        result.append([cluster, statistics.mode(getattr(dataframe[dataframe[df_cluster_name] == cluster], df_feature_name))])
    
    return result

def id_mode_of_clusters(dataframe, df_id_name, df_feature_name, df_cluster_name):
    id_list = list(set(getattr(dataframe, df_id_name)))
    id_cluster_list = id_clusters_list(dataframe, id_list, df_id_name, df_cluster_name)
    cluster_mode = cluster_id_mode_list(dataframe, df_feature_name, df_cluster_name)
    indexing_list = [item[0] for item in cluster_mode]
    count = []
    result = []

    for id_elem, clusters in id_cluster_list:
        for cluster in clusters:
            count.append(cluster_mode[indexing_list.index(cluster)][1])
        result.append([id_elem, statistics.mode(count)])
        count.clear()

    return result

def new_method_mode(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    id_mode = id_mode_of_clusters(dataframe, df_id_name, df_feature_name, df_cluster_name)
    indexing = [item[0] for item in id_mode]
    result = []
    
    for missing_id_elem, missing_time_elem in missing_rows:
        result.append([missing_id_elem, missing_time_elem, id_mode[indexing.index(missing_id_elem)][1]])
    
    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "new_method_mode"])


#Pre_And_Post_Feature_Analysis

def id_mfcm_list_ppa(missing_id_list, id_cluster_list):
    most_frequent_cluster_member = []
    mfcm_ids = []
    current_count = 0
    biggest_count = 0

    for missing_id_elem in missing_id_list:
        for id_elem, clusters_elem in id_cluster_list:
            if missing_id_elem != id_elem:
                current_count = len(set(id_cluster_list[missing_id_list.index(missing_id_elem)][1]).intersection(clusters_elem))
                if current_count == biggest_count:
                    mfcm_ids.append(id_elem)
                if current_count > biggest_count:
                    biggest_count = current_count
                    mfcm_ids.clear()
                    mfcm_ids.append(id_elem)
        most_frequent_cluster_member.append([missing_id_elem, list(mfcm_ids)])
        mfcm_ids.clear()
        biggest_count = 0

    return most_frequent_cluster_member

def pre_features(dataframe, list_of_ids, time, df_id_name, df_time_name, df_feature_name):
    df_times = list(set(getattr(dataframe, df_time_name)))
    df_times.reverse()
    index = df_times.index(time)
    pre_features = []
    
    for id_elem in list_of_ids:
        for time_elem in df_times[index+1:]:
            row = dataframe[(dataframe[df_id_name] == id_elem) & (dataframe[df_time_name] == time_elem)]
            if not row.empty:
                if not getattr(row, df_feature_name).values is None:
                    pre_features.append(getattr(row, df_feature_name).values[0])
                    break
    
    return pre_features


def post_features(dataframe, list_of_ids, time, df_id_name, df_time_name, df_feature_name):
    df_times = list(set(getattr(dataframe, df_time_name)))
    index = df_times.index(time)
    post_features = []
    
    for id_elem in list_of_ids:
        for time_elem in df_times[index+1:]:
            row = dataframe[(dataframe[df_id_name] == id_elem) & (dataframe[df_time_name] == time_elem)]
            if not row.empty:
                if not getattr(row, df_feature_name).values is None:
                    post_features.append(getattr(row, df_feature_name).values[0])
                    break
    
    return post_features


def pre_and_post_features(dataframe, list_of_ids, time, df_id_name, df_time_name, df_feature_name):
    result = []
    
    result.extend(pre_features(dataframe, list_of_ids, time, df_id_name, df_time_name, df_feature_name))
    result.extend(post_features(dataframe, list_of_ids, time, df_id_name, df_time_name, df_feature_name))

    return result


def pre_and_post_clustering_analysis(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    missing_id_list = sorted(list(set([item[0] for item in missing_rows])))
    id_cluster_list = id_clusters_list(dataframe, missing_id_list, df_id_name, df_cluster_name)
    mfcm = id_mfcm_list_ppa(missing_id_list, id_cluster_list)
    result = []
    
    for id_elem, time_elem in missing_rows:
        for index in range(len(mfcm)):
            if id_elem == mfcm[index][0]:
                result.append([id_elem, time_elem, statistics.mean(pre_and_post_features(dataframe, list(flatten(mfcm[index])), time_elem, df_id_name, df_time_name, df_feature_name))])
    
    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "ppa"])


def mean_timestemp(dataframe, df_id_name, df_time_name, df_feature_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    result = []

    for id_elem, time_elem in missing_rows:
        result.append([id_elem, time_elem, statistics.mean(getattr(dataframe[dataframe[df_time_name] == time_elem], df_feature_name))])

    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "mean_timestemp"])

def median_timestemp(dataframe, df_id_name, df_time_name, df_feature_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    result = []

    for id_elem, time_elem in missing_rows:
        result.append([id_elem, time_elem, statistics.median(getattr(dataframe[dataframe[df_time_name] == time_elem], df_feature_name))])

    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "median_timestemp"])

def mean_timeseries(dataframe, df_id_name, df_time_name, df_feature_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    result = []

    for id_elem, time_elem in missing_rows:
        result.append([id_elem, time_elem, statistics.mean(getattr(dataframe[dataframe[df_id_name] == id_elem], df_feature_name))])

    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "mean_timeseries"])

def median_timeseries(dataframe, df_id_name, df_time_name, df_feature_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    result = []

    for id_elem, time_elem in missing_rows:
        result.append([id_elem, time_elem, statistics.median(getattr(dataframe[dataframe[df_id_name] == id_elem], df_feature_name))])

    return pd.DataFrame(np.array(result), columns=[df_id_name, df_time_name, "median_timeseries"])


#berechnet bestes Clustering basierend auf CLOSE Score

def calc_best_rating(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    calc_rating_df = dataframe
    rating_list = [0]
    for x in range(2, 7):
        for y in range(10, 41):
            test_db1 = DBSCAN(eps = y/100, min_samples = x).fit(calc_rating_df[[df_time_name, df_feature_name]])
            calc_rating_df[df_cluster_name] = test_db1.labels_

            rater = CLOSE(calc_rating_df[[df_id_name, df_time_name, df_cluster_name, df_feature_name]], 'exploit', 2, jaccard=True, weighting=True)
            rating = rater.rate_time_clustering()
            
            if rating > rating_list[0]:
                rating_list.clear()
                rating_list.append(rating)
                rating_list.append(y/100)
                rating_list.append(x)
    
    return rating_list


def avg_distance(series_A, series_B, na_indexes):

    result = series_A.where(series_A.index.isin(na_indexes)).dropna() - series_B.where(series_B.index.isin(na_indexes)).dropna()

    return statistics.median(np.absolute(result))