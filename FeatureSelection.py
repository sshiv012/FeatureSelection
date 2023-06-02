import math

# Find Euclidean Distance between two points
def euclidean_distance(p1, p2):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
    return distance

# Find Manhattan Distance between two points
def manhattan_distance(p1, p2):
    distance = sum([abs(a - b) for a, b in zip(p1, p2)])
    return distance

def nearest_neighbors(data):
    classes = [float(row[0]) for row in data]
    features = [row[1:] for row in data]
    predictions = []

    for i in range(len(features)):
        test_feature = features[i]
        test_class = classes[i]

        distances = []
        labels = []

        for j in range(len(features)):
            if i != j:
                distance = euclidean_distance(features[j], test_feature)
                #distance = manhattan_distance(features[j], test_feature)
                distances.append(distance)
                labels.append(classes[j])

        min_distance_index = distances.index(min(distances))
        predicted_label = labels[min_distance_index]
        predictions.append(predicted_label)
    
    correct_predictions = sum([1 for true, pred in zip(classes, predictions) if true == pred])
    accuracy = correct_predictions / len(classes)
    return accuracy



def main():
    print("Welcome to Suryaa/Bhavya's Feature Selection program.")
    print("Type in the name of the file to test : ")
    file_name = input()
    print("Type the number of the algorithm you want to run")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    algo_number = int(input())
    if algo_number<=0 or algo_number>2:
        print("Defaulting to Forward Selection, as o")
        algo_number=1
    file = open(file_name, 'r')
    first_line = file.readline()
    columns = first_line.split()
    num_records = sum(1 for _ in file)
    
    num_features = len(columns) - 1
    file.close()

    print(f'This dataset has {num_features} features(not including class attribute), with {num_records+1} instances')

    normalized_data = normalize_data(file_name)
    accuracy = nearest_neighbors(normalized_data)
    print(f'Running nearest neighbor with all {num_features} features, using \"leaving-one-out\" evaluation, we get an accuracy of {accuracy*100}%')

#0/1 Normalization
def normalize_data(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    normalized_data = []

    for line in lines:
        columns= line.strip().split()
        features = [float(column) for column in columns[1:]]
        min_val = min(features)
        max_val = max(features)
        normalized_features = [(float(x) - min_val) / (max_val - min_val) for x in features]
        normalized_line = [columns[0]] + normalized_features
        normalized_data.append(normalized_line)
    return normalized_data
      
if __name__=="__main__":
    main()  