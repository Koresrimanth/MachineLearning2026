
def drop_unused_columns(df,coluns_to_drop):
   
    df = df.drop(
        columns=[coluns_to_drop],
        errors="ignore"
    )

    return df