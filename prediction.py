# @hidden_cell
from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.helpers import S3Connection, S3Location

training_data_reference = [DataConnection(
    connection=S3Connection(
        api_key='K47y4vRhVc2PvgAPTJMqzmtOtpH7ggd-NvdZ__EO6VJn',
        auth_endpoint='https://iam.bluemix.net/oidc/token/',
        endpoint_url='https://s3.eu-geo.objectstorage.softlayer.net'
    ),
    location=S3Location(
        bucket='hearinglossprediction-donotdelete-pr-tuqv5iquivlabu',
        path='sound.csv'
    )),
]
training_result_reference = DataConnection(
    connection=S3Connection(
        api_key='K47y4vRhVc2PvgAPTJMqzmtOtpH7ggd-NvdZ__EO6VJn',
        auth_endpoint='https://iam.bluemix.net/oidc/token/',
        endpoint_url='https://s3.eu-geo.objectstorage.softlayer.net'
    ),
    location=S3Location(
        bucket='hearinglossprediction-donotdelete-pr-tuqv5iquivlabu',
        path='auto_ml/7aac07bf-f6c2-4df0-a467-59f6a462ded9/wml_data/e3cd4099-9f3a-4ece-af29-d6982ebea37b/data/automl',
        model_location='auto_ml/7aac07bf-f6c2-4df0-a467-59f6a462ded9/wml_data/e3cd4099-9f3a-4ece-af29-d6982ebea37b/data/automl/hpo_c_output/Pipeline9/model.pickle',
        training_status='auto_ml/7aac07bf-f6c2-4df0-a467-59f6a462ded9/wml_data/e3cd4099-9f3a-4ece-af29-d6982ebea37b/training-status.json'
    ))

experiment_metadata = dict(
    prediction_type='regression',
    prediction_column='Hearing loss threshold',
    holdout_size=0.1,
    scoring='neg_root_mean_squared_error',
    deployment_url='https://eu-gb.ml.cloud.ibm.com',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=4,
    training_data_reference=training_data_reference,
    training_result_reference=training_result_reference,
    project_id='0125b23b-4d6f-466c-b846-2b957461c6da',
    drop_duplicates=True
)

df = training_data_reference[0].read(csv_separator=experiment_metadata['csv_separator'])
df.dropna('rows', how='any', subset=[experiment_metadata['prediction_column']], inplace=True)

from sklearn.model_selection import train_test_split

df.drop_duplicates(inplace=True)
X = df.drop([experiment_metadata['prediction_column']], axis=1).values
y = df[experiment_metadata['prediction_column']].values

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=experiment_metadata['holdout_size'],
                                                    random_state=experiment_metadata['random_state'])

from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import CompressStrings
from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues
from autoai_libs.transformers.exportable import boolean2float
from autoai_libs.transformers.exportable import CatImputer
from autoai_libs.transformers.exportable import CatEncoder
import numpy as np
from autoai_libs.transformers.exportable import float32_transform
from sklearn.pipeline import make_pipeline
from autoai_libs.transformers.exportable import FloatStr2Float
from autoai_libs.transformers.exportable import NumImputer
from autoai_libs.transformers.exportable import OptStandardScaler
from sklearn.pipeline import make_union
from autoai_libs.transformers.exportable import NumpyPermuteArray
from autoai_libs.cognito.transforms.transform_utils import TA1
import autoai_libs.utils.fc_methods
from autoai_libs.cognito.transforms.transform_utils import FS1
from autoai_libs.cognito.transforms.transform_utils import TA2
from sklearn.ensemble import RandomForestRegressor

numpy_column_selector_0 = NumpyColumnSelector(columns=[3, 4, 5, 6, 7])
compress_strings = CompressStrings(
    compress_type="hash",
    dtypes_list=["int_num", "int_num", "int_num", "int_num", "char_str"],
    missing_values_reference_list=["", "-", "?", float("nan")],
    misslist_list=[[], [], [], [], []],
)
numpy_replace_missing_values_0 = NumpyReplaceMissingValues(
    missing_values=[], filling_values=float("nan")
)
numpy_replace_unknown_values = NumpyReplaceUnknownValues(
    filling_values=float("nan"),
    filling_values_list=[
        float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
    ],
    missing_values_reference_list=["", "-", "?", float("nan")],
)
cat_imputer = CatImputer(
    strategy="most_frequent",
    missing_values=float("nan"),
    sklearn_version_family="23",
)
cat_encoder = CatEncoder(
    encoding="ordinal",
    categories="auto",
    dtype=np.float64,
    handle_unknown="error",
    sklearn_version_family="23",
)
pipeline_0 = make_pipeline(
    numpy_column_selector_0,
    compress_strings,
    numpy_replace_missing_values_0,
    numpy_replace_unknown_values,
    boolean2float(),
    cat_imputer,
    cat_encoder,
    float32_transform(),
)
numpy_column_selector_1 = NumpyColumnSelector(columns=[0, 1, 2])
float_str2_float = FloatStr2Float(
    dtypes_list=["int_num", "int_num", "int_num"],
    missing_values_reference_list=[],
)
numpy_replace_missing_values_1 = NumpyReplaceMissingValues(
    missing_values=[], filling_values=float("nan")
)
num_imputer = NumImputer(strategy="median", missing_values=float("nan"))
opt_standard_scaler = OptStandardScaler(
    num_scaler_copy=None,
    num_scaler_with_mean=None,
    num_scaler_with_std=None,
    use_scaler_flag=False,
)
pipeline_1 = make_pipeline(
    numpy_column_selector_1,
    float_str2_float,
    numpy_replace_missing_values_1,
    num_imputer,
    opt_standard_scaler,
    float32_transform(),
)
union = make_union(pipeline_0, pipeline_1)
numpy_permute_array = NumpyPermuteArray(
    axis=0, permutation_indices=[3, 4, 5, 6, 7, 0, 1, 2]
)
ta1 = TA1(
    fun=np.log,
    name="log",
    datatypes=["numeric"],
    feat_constraints=[
        autoai_libs.utils.fc_methods.is_positive,
        autoai_libs.utils.fc_methods.is_not_categorical,
    ],
    col_names=[
        "Age", "Carrer length", "Noise exposure level", "threshold min",
        "threshold max", "Noise exposure max", "Noise exposure min",
        "Smoking",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"),
    ],
)
fs1_0 = FS1(
    cols_ids_must_keep=range(0, 8),
    additional_col_count_to_keep=8,
    ptype="regression",
)
ta2 = TA2(
    fun=np.multiply,
    name="product",
    datatypes1=[
        "intc", "intp", "int_", "uint8", "uint16", "uint32", "uint64", "int8",
        "int16", "int32", "int64", "short", "long", "longlong", "float16",
        "float32", "float64",
    ],
    feat_constraints1=[autoai_libs.utils.fc_methods.is_not_categorical],
    datatypes2=[
        "intc", "intp", "int_", "uint8", "uint16", "uint32", "uint64", "int8",
        "int16", "int32", "int64", "short", "long", "longlong", "float16",
        "float32", "float64",
    ],
    feat_constraints2=[autoai_libs.utils.fc_methods.is_not_categorical],
    col_names=[
        "Age", "Carrer length", "Noise exposure level", "threshold min",
        "threshold max", "Noise exposure max", "Noise exposure min",
        "Smoking", "log(Age)", "log(Carrer length)",
        "log(Noise exposure level)",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"),
    ],
)
fs1_1 = FS1(
    cols_ids_must_keep=range(0, 8),
    additional_col_count_to_keep=8,
    ptype="regression",
)
random_forest_regressor = RandomForestRegressor(
    criterion="friedman_mse",
    max_depth=3,
    max_features=0.9945334984207548,
    min_samples_leaf=3,
    n_estimators=53,
    n_jobs=4,
    random_state=33,
)

pipeline = make_pipeline(
    union,
    numpy_permute_array,
    ta1,
    fs1_0,
    ta2,
    fs1_1,
    random_forest_regressor,
)

from sklearn.metrics import get_scorer

scorer = get_scorer(experiment_metadata['scoring'])

pipeline.fit(train_X, train_y)

score = scorer(pipeline, test_X, test_y)