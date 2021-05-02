import unittest
import logging
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient
from models_preparation import create_tensorflow_model_data


class TestWMLClientWithTensorflow(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    cos_resource_instance_id = None
    scoring_data = None
    logger = logging.getLogger(__name__)
    space_name = 'tests_sdk_space'
    model_path = 'svt/artifacts/tf_iris_model'

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)
        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestWMLClientWithTensorflow.space_id = get_space_id(self.wml_client, self.space_name,
                                                 cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_00b_create_tensorflow_model(self):
        res = create_tensorflow_model_data()
        TestWMLClientWithTensorflow.scoring_data = res['test_data']['x_test']

        import tensorflow as tf
        tf.saved_model.save(res['model'], self.model_path)

    def test_02_publish_local_model_in_repository(self):
        TestWMLClientWithTensorflow.logger.info("Saving trained model in repo ...")
        TestWMLClientWithTensorflow.logger.debug(self.model_path)

        self.wml_client.repository.ModelMetaNames.show()

        sw_spec_id = self.wml_client.software_specifications.get_id_by_name('tensorflow_2.4-py3.7')

        model_meta_props = {self.wml_client.repository.ModelMetaNames.NAME: "LOCALLY created agaricus prediction model",
                       self.wml_client.repository.ModelMetaNames.TYPE: "tensorflow_2.4",
                       self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                            }
        published_model_details = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        TestWMLClientWithTensorflow.model_uid = self.wml_client.repository.get_model_uid(published_model_details)
        TestWMLClientWithTensorflow.logger.info("Published model ID:" + str(TestWMLClientWithTensorflow.model_uid))
        self.assertIsNotNone(TestWMLClientWithTensorflow.model_uid)

    def test_04_load_model(self):
        print(TestWMLClientWithTensorflow.model_uid)
        TestWMLClientWithTensorflow.logger.info("Load model from repository: {}".format(TestWMLClientWithTensorflow.model_uid))
        self.tf_model = self.wml_client.repository.load(TestWMLClientWithTensorflow.model_uid)
        TestWMLClientWithTensorflow.logger.debug("TF type: {}".format(type(self.tf_model)))

    def test_05_create_deployment(self):
        TestWMLClientWithTensorflow.logger.info("Create deployment")
        deployment_details = self.wml_client.deployments.create(artifact_uid=TestWMLClientWithTensorflow.model_uid,
                                                            meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",
                                                                        self.wml_client.deployments.ConfigurationMetaNames.ONLINE: {}})


        TestWMLClientWithTensorflow.deployment_uid = self.wml_client.deployments.get_uid(deployment_details)
        TestWMLClientWithTensorflow.scoring_url = self.wml_client.deployments.get_scoring_href(deployment_details)
        self.assertTrue('online' in str(deployment_details))

    def test_06_scoring(self):
        TestWMLClientWithTensorflow.logger.info("Score model")

        scoring_payload = {
            'input_data': [
                {
                    'values': self.scoring_data.tolist()
                }
            ]
        }
        self.wml_client.deployments.ScoringMetaNames.show()
        scores = self.wml_client.deployments.score(TestWMLClientWithTensorflow.deployment_uid, meta_props=scoring_payload)
        self.assertIsNotNone(scores)

    def test_07_delete_deployment(self):
        TestWMLClientWithTensorflow.logger.info("Delete deployment")
        self.wml_client.deployments.delete(TestWMLClientWithTensorflow.deployment_uid)

    def test_08_delete_model(self):
        TestWMLClientWithTensorflow.logger.info("Delete model")
        self.wml_client.repository.delete(TestWMLClientWithTensorflow.model_uid)


if __name__ == '__main__':
    unittest.main()
