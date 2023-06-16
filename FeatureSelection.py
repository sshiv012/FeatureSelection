import math
import csv
import time
import random

# Find Euclidean Distance between two points
def euclidean_distance(p1, p2):
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    return distance

# Find Manhattan Distance between two points
# Not used in this program, only for evaluation
def manhattan_distance(p1, p2):
    distance = sum([abs(a - b) for a, b in zip(p1, p2)])
    return distance

def nearest_neighbors(data, current_feature_set, best_correct_predictions=0):
    classes = [row[0] for row in data]
    instances = [[row[1][col-1] for col in current_feature_set] for row in data]

    correct_predictions = 0
    incorrect_predictions = 0

    for i in range(len(instances)):
        test_feature = instances[i]
        test_class = classes[i]
        
        distances = []
        labels = []

        for j in range(len(instances)):
            if i != j:
                distance = euclidean_distance(test_feature, instances[j])
                #distance = manhattan_distance(features[j], test_feature)
                distances.append(distance)
                labels.append(classes[j])

        min_distance_index = distances.index(min(distances))
        predicted_label = labels[min_distance_index]
        if test_class == predicted_label:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
        
        if len(instances) - incorrect_predictions < best_correct_predictions:
            return (-1, 0)

    accuracy = round((correct_predictions / len(instances)), 3)

    return (accuracy, correct_predictions)

def forward_search(data, num_features, early_stop=False):
    current_set_of_features = [] # empty set
    best_set_of_features = [] # empty set
    total_best_accuracy = 0.0
    print(f'Beginning forward search.')

    for i in range(1, num_features + 1):
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0.0
        best_correct_predictions = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:
                accuracy, correct_predictions = nearest_neighbors(data, current_set_of_features + [k], best_correct_predictions)
                if accuracy == -1:
                    print(f'Using feature(s) {{{current_set_of_features + [k]}}} accuracy < {best_accuracy_so_far*100}%')
                    continue

                print(f'Using feature(s) {{{current_set_of_features + [k]}}} accuracy is {accuracy*100}%')
                
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
                    if correct_predictions > best_correct_predictions:
                        best_correct_predictions = correct_predictions


        current_set_of_features.append(feature_to_add_at_this_level)

        if best_accuracy_so_far > total_best_accuracy:
            print('here')
            total_best_accuracy = best_accuracy_so_far
            best_set_of_features = current_set_of_features[:]
            consecutive_decreases = 0
        else:
            print(f'(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
            consecutive_decreases += 1
        print(f'Feature set {current_set_of_features} was best, accuracy is {best_accuracy_so_far*100}%')

        if early_stop and consecutive_decreases >= 3:
            print(f'Accuracy has decreased for five consecutive steps. Stopping early.')
            break

    print(f'Finished search!! The best feature subset is {best_set_of_features}, which has an accuracy of {total_best_accuracy*100}%')

    return best_set_of_features, total_best_accuracy

def backward_elimination(data, num_features):
     current_feature_set = set(i for i in range(1, num_features + 1)) #Entire set of features
     best_feature_set = current_feature_set.copy() #Initialization
     total_best_accuracy = 0.0
     
     print(f'Beginning search using Backward Elimination')
     for i in range(num_features, 0, -1):
        print(f'On the {i}th level of the search tree.')
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0.0
        best_correct_predictions = 0

        for k in current_feature_set:
            print(f'Considering removing the {k}th feature')
            feature_subset = current_feature_set-{k}
            accuracy, correct_predictions = nearest_neighbors(data, feature_subset, best_correct_predictions)
            if accuracy == -1:
                    print(f'Using feature(s) {{{", ".join(str(f) for f in feature_subset)}}} accuracy < {best_accuracy_so_far*100}%')
                    continue
            print(f'Using feature(s) {{{", ".join(str(f) for f in feature_subset)}}} accuracy is {accuracy*100}%')

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = k
                if correct_predictions > best_correct_predictions:
                    best_correct_predictions = correct_predictions

        current_feature_set.remove(feature_to_remove_at_this_level)
        
        if best_accuracy_so_far > total_best_accuracy:
            total_best_accuracy = best_accuracy_so_far
            best_feature_set = current_feature_set.copy()
            print('(Accuracy has increased, we have escaped a local maxima!)')
        else:
            print('(No improvement this path)')
        print(f'Feature set {{{", ".join(str(f) for f in current_feature_set)}}} was best, accuracy is {best_accuracy_so_far * 100}%')

     print(f'Finished search!! The best feature subset is {{{", ".join(str(f) for f in best_feature_set)}}}, which has an accuracy of {total_best_accuracy * 100}%')
     return best_feature_set, total_best_accuracy
    
def main():
    print("Welcome to Suryaa/Bhavya's Feature Selection program.")
    print("Type in the name of the file to test, type default to run on Wisconsin Breast Cancer Dataset : ")
    file_name = input()

    print("Type the number of the algorithm you want to run")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    algo_number = int(input())
    if algo_number<=0 or algo_number>2:
         print("Defaulting to Forward Selection, as you have entered an invalid number")
         algo_number=1
    if file_name == "default":
        file_name = "data.csv"
        csv_data = []
        with open(file_name, 'r') as file:
            reader = csv.DictReader(file)
            num_features = len(reader.fieldnames) - 2  # exclude 'id' and 'diagnosis' columns
            num_records = 0
            for row in reader:
                csv_data.append(row)
                num_records += 1
    else:
        with open(file_name, 'r') as file:
            file_data = file.readlines()
            num_features = len(file_data[0].split()) - 1 # exclude 1st column (classes)
            num_records = len(file_data) # count from 2nd line

    print(num_records)
    print(num_features)
    print(f'This dataset has {num_features} features(not including class attribute), with {num_records} instances')

    # normalized_data = normalize(data)
    data = []

    if file_name == "data.csv":
        for row in csv_data:
            # classes.append(float(1) if row['diagnosis'] == 'M' else float(2))
            instance = []
            for feature in reader.fieldnames[2:]:
                value = row[feature]
                if value is None:
                    print(feature)
                    print(row)
                instance.append(float(value) if value is not None else None)
            # instances.append(instance)
            data.append([float(1) if row['diagnosis'] == 'M' else float(2), instance])
    else:
        for line in file_data:
            row = line.strip().split()
            data.append([float(row[0]), [float(i) for i in row[1:]]])
            # classes.append(float(row[0]))
            # instances.append([float(i) for i in row[1:]])
    
    # data = [classes, instances]
    if len(data) > 10000:
        sample_size = len(data) // 2
        random.shuffle(data)
        data = data[:sample_size]
    print(len(data))
    accuracy, _ = nearest_neighbors(data, [i for i in range(1, num_features + 1)])
    print(f'Running nearest neighbor with all {num_features} features, using \"leaving-one-out\" evaluation, we get an accuracy of {accuracy*100}%')

    start_time = time.time()
    if algo_number == 1:
        print("Do you want to stop early if there are consecutive decreases in accuracy (OPTIMIZATION)? (y/n)")
        stop_early = input()
        if stop_early == 'y':
            final_feature_set = forward_search(data, num_features, True)
        else:
            final_feature_set = forward_search(data, num_features)
        end_time = time.time() - start_time
        print(f'Finished forward search in {end_time} seconds.')
    else:
        final_feature_set = backward_elimination(data, num_features)
        end_time = time.time() - start_time
        print(f'Finished backward elimination search in {end_time} seconds.')
    print(f'Final feature subset: {final_feature_set}')

#0/1 Normalization
def normalize(data):
    print(f'lengh of data: {len(data)}')
    normalized_data = []

    for line in data:
        columns = line.strip().split()
        features = [float(column) for column in columns[1:]]
        min_val = min(features)
        max_val = max(features)
        normalized_features = [(float(x) - min_val) / (max_val - min_val) for x in features]
        normalized_line = [columns[0]] + normalized_features
        normalized_data.append(normalized_line)
    return normalized_data
      
if __name__=="__main__":
    main()  