from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from preprocess.encoding import CustomOneHotEncoding
from preprocess.date_converter import DateConverter
from preprocess.features_selection import BestFeatures
from preprocess.scaler import CustomScaler
import pandas as pd

def basic_preprocess(df, target):
    dropable_cols           = ['Unnamed: 0', 'page_url', 'property_id']
    target_encoding_cols    = ['location', 'agency', 'agent']
    oh_encoding_cols        = ['property_type', 'city', 'province_name', 'purpose']
    date_encoding           = ['date_added']

    df = df.drop(dropable_cols, axis=1)
    df = df.drop_duplicates()
    df = df.drop(df.query("agency != agency and agent != agent").index)
    df = df.drop(155597)
    num_cols = df.drop(target,axis=1).select_dtypes(exclude="object").columns.tolist()

    [df_train, df_test] = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    df_test = df_test._append(df_train.iloc[0])

    pipe = Pipeline(steps=[
        ("target_encoding", TargetEncoder(verbose=10,cols=target_encoding_cols)),
        ("oh_encoding", CustomOneHotEncoding(attributes=oh_encoding_cols)),
        ("date_converter", DateConverter(date_encoding)),
        ("scaler", CustomScaler(num_cols)),
        ("features_selection", BestFeatures(0.02, False)),
    ])

    print(df_train.iloc[0].to_frame().T)
    print(df_test.iloc[-1].to_frame().T)

    X_train =  pd.DataFrame(pipe.fit_transform(df_train.drop(target, axis=1), df_train[target]), index=df_train.index)
    X_test = pd.DataFrame(pipe.transform(df_test.drop(target, axis=1)), index=df_test.index)

    df_train = pd.concat([X_train, df_train[target] ], axis=1)
    df_test = pd.concat([X_test, df_test[target] ], axis=1)

    print(df_train.iloc[0].to_frame().T)
    print(df_test.iloc[-1].to_frame().T)


    print(df_train.head(5))
    print(df_test.head(5))


    return [df_train, df_test]
