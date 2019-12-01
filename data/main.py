import pandas as pd
import numpy as np
import os



def main():
    
    tst = pd.read_csv('d:/UNI/4ano/IA/Code/data/iris_categorical/bezdekIris.data',delimiter='\n',header=None)
    df = pd.DataFrame(tst[0].str.split(',').tolist())
    df.columns = ['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm','class']

    # dataset de treino com metade das samples das classes setosa e versicolor
    df = df.loc[(df['class']=='Iris-setosa') | (df['class']=='Iris-versicolor')]

   
    #X = df.loc[:,['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm']]
    #Y = df.loc[:,['class']]




    #df['class'] = pd.factorize(df['class'])[0] + 1

    #X['class'] = Y = pd.factorize(df['class'])[0] + 1
    
    df["class"] = df["class"].astype('category')

    class_cat = df["class"].cat.codes

    del df["class"]

    df.insert(0,"class",class_cat)

    

    df.to_csv('d:/UNI/4ano/IA/Code/data/iris_categorical/df.txt',header=True,index=False)

    print(df)



if __name__== "__main__":
  main()