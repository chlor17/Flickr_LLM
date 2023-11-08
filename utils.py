import yaml

def read_yaml(file_path = "config.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    

def create_df_from_photo(photo):
    col_list = ['image_name',
                'photo_id',
                'user_id',
                'destination_path',
                'image_path',
                'url',
                'caption',
                'story',
                'aperture',
                'exposuretime',
                'iso',
                'createdate']

    df = pd.DataFrame(columns=col_list)

    row = {}
    for col in col_list:
        row[col]  = getattr(photo, col)
    df = pd.concat([df, pd.DataFrame.from_dict(row)], ignore_index=True)
    return spark.createDataFrame(df)