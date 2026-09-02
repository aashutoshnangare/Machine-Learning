from sklearn.datasets import load_iris

def main():
    print("Iris Classification Case Study")

    Dataset = load_iris()

    #Meta data of Dataset
    print("Independent variables are :")
    print(Dataset.feature_names)
    print("length of independent variable is : ",len(Dataset.feature_names))

    print("Dependent variables are :")
    print(Dataset.target_names)
    print("length of dependent variable is : ",len(Dataset.target_names))


if __name__ == "__main__":
    main()
