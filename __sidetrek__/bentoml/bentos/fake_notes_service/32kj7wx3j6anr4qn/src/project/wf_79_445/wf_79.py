import sklearn
import os
import typing
from flytekit import task, workflow, Resources
import sidetrek
from project.wf_79_445.main import Hyperparameters
from project.wf_79_445.main import create_dataframe
from project.wf_79_445.main import split_dataset
from project.wf_79_445.main import train_model

@task(requests=Resources(cpu="2",mem="1Gi"),limits=Resources(cpu="2",mem="1Gi"),retries=3)
def dataset_test_org_fake_notes_data()->sidetrek.types.dataset.SidetrekDataset:
	return sidetrek.dataset.build_dataset(io="upload",source="s3://sidetrek-datasets/test-org/fake-notes-data")



_wf_outputs=typing.NamedTuple("WfOutputs",train_model_0=sklearn.ensemble._gb.GradientBoostingClassifier)
@workflow
def wf_79(_wf_args:Hyperparameters)->_wf_outputs:
	dataset_test_org_fake_notes_data_o0_=dataset_test_org_fake_notes_data()
	create_dataframe_o0_=create_dataframe(ds=dataset_test_org_fake_notes_data_o0_)
	split_dataset_o0_,split_dataset_o1_,split_dataset_o2_,split_dataset_o3_=split_dataset(df=create_dataframe_o0_,hp=_wf_args)
	train_model_o0_=train_model(X_train=split_dataset_o0_,y_train=split_dataset_o2_,hp=_wf_args)
	return _wf_outputs(train_model_o0_)