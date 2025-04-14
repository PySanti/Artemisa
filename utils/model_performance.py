from sklearn.metrics import mean_absolute_error, r2_score
def model_performance(estimator, best_hp, df_train, df_test, target, label):
    model = estimator(**best_hp)
    model.fit(df_train.drop(target, axis=1), df_train[target])

    train_predict = model.predict(df_train.drop(target, axis=1))
    test_predict = model.predict(df_test.drop(target, axis=1))

    
    print(f"Rendimiento de {label} para train : {mean_absolute_error(df_train[target], train_predict):.3f}")
    print(f"Rendimiento de {label} para test : {mean_absolute_error(df_test[target], test_predict):.3f}")

    print(f"r2 de {label} para train : {r2_score(df_train[target], train_predict):.3f}")
    print(f"r2 de {label} para test : {r2_score(df_test[target], test_predict):.3f}")

    print(best_hp)
    print("__________________________")
