import optuna
study = optuna.load_study(
    study_name="sugarcane_efficientnet_study",
    storage="sqlite:///sugarcane_optuna_study.db"
)
df = study.trials_dataframe()
pd_cols = [c for c in ["number", "state", "value", "datetime_start", "datetime_complete",
                       "params_learning_rate", "params_dropout_rate",
                       "params_optimizer", "params_dense_units"] if c in df.columns]
print(df[pd_cols].to_string())
print(df["state"].value_counts())