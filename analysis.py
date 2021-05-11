import pandas as pd
import matplotlib.pyplot as plt

def analyzeNum():
    data = pd.read_csv('cid_data.csv')
    print(data['Unnamed: 0']) 
    dlength = sorted(data['data_length'], reverse = True)
    
    plt.plot(data['Unnamed: 0'], dlength)
    plt.xlabel('client')
    plt.ylabel('data size')
    plt.show()


if __name__ == "__main__":
    analyzeNum()

