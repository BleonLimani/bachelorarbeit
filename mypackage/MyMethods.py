import warnings
warnings.filterwarnings('ignore')
from pandas.core.common import flatten
import statistics

#Allgemeine Methoden

def get_missing_rows(dataframe, df_id_name, df_time_name):  #df_feature_name
    id_list = list(set(dataframe[df_id_name]))
    time_list = list(set(dataframe[df_time_name]))
    result = []

    for id_elem in id_list:
        for time_elem in time_list:
            if dataframe[(dataframe[df_id_name] == id_elem) & (dataframe[df_time_name] == time_elem)].empty: # | dataframe[(dataframe[df_feature_name] == None)]
                result.append([id_elem, time_elem])
    
    return result  


def id_assignments_list(dataframe, missing_id_list, df_id_name, df_cluster_name):
    id_assignment_list = []
    needed_assignments = []

    for id_elem in missing_id_list:
        needed_assignments = list(set(getattr(dataframe[dataframe[df_id_name] == id_elem], df_cluster_name)))
        if -1 in needed_assignments:
            needed_assignments.remove(-1)
        id_assignment_list.append([id_elem, list(needed_assignments)])
        needed_assignments.clear()

    return id_assignment_list


def id_mfcm_list(missing_id_list, id_assignment_list):
    most_frequent_cluster_member = []
    mfcm_ids = []
    current_count = 0
    biggest_count = 0

    for missing_id_elem in missing_id_list:
        for id_elem, assignments_elem in id_assignment_list:
            if missing_id_elem != id_elem:
                current_count = len(set(id_assignment_list[missing_id_list.index(missing_id_elem)][1]).intersection(assignments_elem))
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

#Most_Frequent_Cluster_Member


def most_frequent_cluster_member(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    missing_id_list = sorted(list(set([item[0] for item in missing_rows])))
    id_assignment_list = id_assignments_list(dataframe, missing_id_list, df_id_name, df_cluster_name)
    mfcm = id_mfcm_list(missing_id_list, id_assignment_list)
    mfcm_id_list = [item[0] for item in mfcm]
    middle = []
    result = []
    
    for missing_id_elem, missing_time_elem in missing_rows:
        for cluster_member in mfcm[mfcm_id_list.index(missing_id_elem)][1]:
            if getattr(dataframe[(dataframe[df_id_name] == cluster_member) & (dataframe[df_time_name] == missing_time_elem)], df_feature_name).values:
                middle.append(getattr(dataframe[(dataframe[df_id_name] == cluster_member) & (dataframe[df_time_name] == missing_time_elem)], df_feature_name).values)
        if middle:
            result.append([missing_id_elem, missing_time_elem, list(statistics.median(middle))])
            middle.clear()
        else:
            result.append([missing_id_elem, missing_time_elem, ["-----------------"]])
        
    return result


#Pre_And_Post_Feature_Analysis

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
                    pre_features.append(getattr(row, df_feature_name).values)
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
                    post_features.append(getattr(row, df_feature_name).values)
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
    id_assignment_list = id_assignments_list(dataframe, missing_id_list, df_id_name, df_cluster_name)
    mfcm = id_mfcm_list(missing_id_list, id_assignment_list)
    result = []
    
    for id_elem, time_elem in missing_rows:
        for index in range(len(mfcm)):
            if id_elem == mfcm[index][0]:
                result.append([id_elem, time_elem, statistics.median(pre_and_post_features(dataframe, list(flatten(mfcm[index])), time_elem, df_id_name, df_time_name, df_feature_name)).tolist()])
    
    return result


#Median_of_All

def median_of_clusterings_test(dataframe, df_time_name, df_feature_name, df_cluster_name):
    middle = []
    result = []
    assignments_list = list(set(getattr(dataframe, df_cluster_name).values))
    
    for cluster_elem in assignments_list:
        if cluster_elem != -1:
            for feature_elem in list(set(getattr(dataframe[dataframe[df_cluster_name] == cluster_elem], df_feature_name).values)):
                if feature_elem:
                    middle.append(feature_elem)
            result.append([cluster_elem, statistics.median(middle)])
            middle.clear()
   
    return result


def median_of_all_test(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    missing_id_list = sorted(list(set([item[0] for item in missing_rows])))
    id_assignment_list = id_assignments_list(dataframe, missing_id_list, df_id_name, df_cluster_name)
    id_assignment_index_list = [item[0] for item in id_assignment_list]
    cluster_median_list = median_of_clusterings_test(dataframe, df_time_name, df_feature_name, df_cluster_name)
    cluster_median_index_list = [item[0] for item in cluster_median_list]
    middle = []
    result = []
    
    for missing_id_elem, missing_time_elem in missing_rows:
        for assignment in id_assignment_list[id_assignment_index_list.index(missing_id_elem)][1]:
            middle.append(cluster_median_list[cluster_median_index_list.index(assignment)][1])
        result.append([missing_id_elem, missing_time_elem, statistics.median(middle)])
        middle.clear()

    
    return result


def median_of_all(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    missing_time_list = sorted(list(set([item[1] for item in missing_rows])))
    medians = median_of_clusterings(dataframe, missing_time_list, df_time_name, df_feature_name, df_cluster_name)
    median_index_list = [item[0] for item in medians]
    result = []
    
    for id_elem, time_elem in missing_rows:
        result.append([id_elem, time_elem, medians[median_index_list.index(time_elem)][1]])
    
    return result

def median_of_clusterings(dataframe, missing_time_list, df_time_name, df_feature_name, df_cluster_name):
    median = []
    median_of_medians = []
    
    for missing_time_elem in missing_time_list:
            assignments = list(set(getattr(dataframe[(dataframe[df_time_name] == missing_time_elem)], df_cluster_name)))
            if -1 in assignments:
                assignments.remove(-1)
            for assign_elem in assignments:
                median.append(statistics.median(getattr(dataframe[(dataframe[df_time_name] == missing_time_elem) & (dataframe[df_cluster_name] == assign_elem)], df_feature_name)))
            median_of_medians.append([missing_time_elem, statistics.median(median)])
            median.clear()

    return median_of_medians


#Mean_of_All


def mean_of_all(dataframe, df_id_name, df_time_name, df_feature_name, df_cluster_name):
    missing_rows = get_missing_rows(dataframe, df_id_name, df_time_name)
    missing_time_list = sorted(list(set([item[1] for item in missing_rows])))
    means = mean_of_clusterings(dataframe, missing_time_list, df_time_name, df_feature_name, df_cluster_name)
    mean_index_list = [item[0] for item in means]
    result = []
    
    for id_elem, time_elem in missing_rows:
        result.append([id_elem, time_elem, means[mean_index_list.index(time_elem)][1]])
    
    return result



def mean_of_clusterings(dataframe, missing_time_list, df_time_name, df_feature_name, df_cluster_name):
    mean = []
    mean_of_means = []
    
    for missing_time_elem in missing_time_list:
            assignments = list(set(getattr(dataframe[(dataframe[df_time_name] == missing_time_elem)], df_cluster_name)))
            if -1 in assignments:
                assignments.remove(-1)
            for assign_elem in assignments:
                mean.append(statistics.mean(getattr(dataframe[(dataframe[df_time_name] == missing_time_elem) & (dataframe[df_cluster_name] == assign_elem)], df_feature_name)))
            mean_of_means.append([missing_time_elem, statistics.mean(mean)])
            mean.clear()

    return mean_of_means
