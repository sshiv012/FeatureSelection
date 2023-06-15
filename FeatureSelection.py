import math

# Find Euclidean Distance between two points
def euclidean_distance(p1, p2):
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    return distance

# Find Manhattan Distance between two points
def manhattan_distance(p1, p2):
    distance = sum([abs(a - b) for a, b in zip(p1, p2)])
    return distance

def nearest_neighbors(data, current_feature_set, feature_to_add=None, best_correct_predictions=0):
    classes = data[0]

    if feature_to_add:
        instances = [[row[col-1] for col in current_feature_set + [feature_to_add]] for row in data[1]]
    else:
        instances = [[row[col-1] for col in current_feature_set] for row in data[1]]

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
            return ("More incorrect", 0)

    accuracy = round((correct_predictions / len(instances)), 3)

    return (accuracy, correct_predictions)

def feature_seaarch(data, num_features):
    current_set_of_features = [] # empty set
    best_set_of_features = [] # empty set
    total_best_accuracy = 0.0
    print(f'Beginning search.')

    for i in range(1, num_features + 1):
        # print(f'On the {i+1}th level of the search tree.')
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0.0
        best_correct_predictions = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:
                # print(f'Considering adding {k}th feature')
                accuracy, correct_predictions = nearest_neighbors(data, current_set_of_features, k, best_correct_predictions)
                if accuracy == "More incorrect":
                    print(f'Using feature(s) {{{current_set_of_features + [k]}}} accuracy < {best_accuracy_so_far}%')
                    continue

                print(f'Using feature(s) {{{current_set_of_features + [k]}}} accuracy is {accuracy}%')
                
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
        else:
            print(f'(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
        print(f'Feature set {current_set_of_features} was best, accuracy is {best_accuracy_so_far}%')

    print(f'Finished search!! The best feature subset is {best_set_of_features}, which has an accuracy of {total_best_accuracy}%')

    return best_set_of_features, total_best_accuracy

def backward_elimination(data, num_features):
     current_feature_set = set(i for i in range(1, num_features + 1)) #Entire set of features
     best_feature_set = current_feature_set.copy() #Initialization
     best_accuracy = 0.0
     print(f'Beginning search using Backward Elimination')
     for i in range(num_features, 0, -1):
        print(f'On the {i}th level of the search tree.')
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0.0
        for k in current_feature_set:
            print(f'Considering removing the {k}th feature')
            feature_subset = current_feature_set-{k}
            accuracy = nearest_neighbors(data, feature_subset)
            print(f'Using feature(s) {{{", ".join(str(f) for f in feature_subset)}}} accuracy is {accuracy * 100}%')

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = k
        current_feature_set.remove(feature_to_remove_at_this_level)
        if best_accuracy_so_far > best_accuracy:
            best_accuracy = best_accuracy_so_far
            best_feature_set = current_feature_set.copy()
            print('(Accuracy has increased, we have escaped a local maxima!)')
        else:
            print('(No improvement this path)')
        print(f'Feature set {{{", ".join(str(f) for f in current_feature_set)}}} was best, accuracy is {best_accuracy_so_far * 100}%')
        print(f'Finished search!! The best feature subset is {{{", ".join(str(f) for f in best_feature_set)}}}, which has an accuracy of {best_accuracy * 100}%')


     print(f'Finished search!! The best feature subset is {{{", ".join(str(f) for f in best_feature_set)}}}, which has an accuracy of {best_accuracy * 100}%')
     return best_feature_set, best_accuracy
    
def main():
    print("Welcome to Suryaa/Bhavya's Feature Selection program.")
    print("Type in the name of the file to test : ")
    # file_name = input()
    file_name = 'CS170_large_Data__21.txt'
    print("Type the number of the algorithm you want to run")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    # algo_number = int(input())
    # if algo_number<=0 or algo_number>2:
    #     print("Defaulting to Forward Selection, as o")
    #     algo_number=1
    with open(file_name, 'r') as file:
        data = file.readlines()
        # first_line = data[0] # 1st line to count total features
        num_features = len(data[0].split()) - 1 # exclude 1st column (classes)
        num_records = len(data) # count from 2nd line

    print(num_records)
    print(num_features)
    print(f'This dataset has {num_features} features(not including class attribute), with {num_records} instances')

    # normalized_data = normalize(data)
    classes = []
    instances = []

    for line in data:
        row = line.strip().split()
        classes.append(float(row[0]))
        instances.append([float(i) for i in row[1:]])
    
    data = [classes, instances]

    accuracy, _ = nearest_neighbors(data, [i for i in range(1, num_features + 1)])
    print(f'Running nearest neighbor with all {num_features} features, using \"leaving-one-out\" evaluation, we get an accuracy of {accuracy*100}%')

    final_feature_set = feature_seaarch(data, num_features)
    print(f'final features: {final_feature_set}')

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