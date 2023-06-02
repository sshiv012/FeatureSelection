
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
