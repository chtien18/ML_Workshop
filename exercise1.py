from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



iris = datasets.load_iris()
x = iris['data']
y=iris['target']

scaler = MinMaxScaler()
scaler1 = StandardScaler()

scaler.fit(x)
scaler1.fit(x)

x_ = scaler.transform(x) # normalized MinMax Scaler
x__ =scaler.transform(x) # normalized Standard Scaler


